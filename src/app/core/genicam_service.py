import os
import time
import base64
import json
import queue
import threading
from collections import deque

from flask_socketio import SocketIO, emit, join_room, leave_room

import numpy as np
import cv2

from harvesters.core import Harvester
import pycuda.driver as cuda

from app.core.config import AppConfig

class GenICamService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenICamService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.app_config = AppConfig()
        self.socketio = None
        self.engine_path = None
        self.camera_index = 0
        self.display_q = queue.Queue(maxsize=5)
        self.infer_q = queue.Queue(maxsize=10)
        self.h = None
        self.ia = None
        self.ctx = None
        self.detector = None
        self.pixel_format = None
        self.vendor = None
        self.capture_thread = None
        self.infer_thread = None
        self.is_capturing = False
        self.is_inferencing = False
        self.done_timestamps = deque(maxlen=60)
        self.conf_threshold = 0.6
        self.client_count = 0
        self.clients_lock = threading.Lock()
        self.display_width = 1280
        self.display_height = 720

    def set_socketio(self, socketio):
        self.socketio = socketio

    def set_engine_path(self, engine_path):
        self.engine_path = engine_path

    def _init_cuda(self):
        cuda.init()
        self.ctx = cuda.Device(0).make_context()

    def _init_camera(self):
        config = self.app_config.get_config()
        cti_path = config.get('cti_file_location')
        
        if not os.path.exists(cti_path):
            raise RuntimeError(f"CTI not found: {cti_path}")

        self.h = Harvester()
        self.h.add_file(cti_path)
        self.h.update()

        if len(self.h.device_info_list) == 0:
            raise RuntimeError("No cameras found via Harvesters/GenTL")

        if self.camera_index >= len(self.h.device_info_list):
            raise RuntimeError(f"Camera index {self.camera_index} not available; {len(self.h.device_info_list)} device(s) detected")

        self.ia = self.h.create(self.camera_index)
        node_map = self.ia.remote_device.node_map
        self.vendor = self.h.device_info_list[self.camera_index].vendor

        node_map.Width.value = node_map.Width.max
        node_map.Height.value = node_map.Height.max

        available = list(node_map.PixelFormat.symbolics)
        priority = ["RGB8", "BGR8", "BayerRG8", "BayerGR8", "BayerBG8", "BayerGB8", "Mono8"]
        selected = None
        for fmt in priority:
            if fmt in available:
                try:
                    node_map.PixelFormat.value = fmt
                    selected = fmt
                    break
                except Exception:
                    continue
        if not selected:
            selected = node_map.PixelFormat.value
        self.pixel_format = selected

    def _init_tensorrt(self):
        if not os.path.exists(self.engine_path):
            raise RuntimeError(f"Engine not found: {self.engine_path}")

        from geni_inference import TensorRTDetector
        self.detector = TensorRTDetector(self.engine_path, max_detections=1000)

    def start(self):
        if self.is_capturing or self.is_inferencing:
            return
        
        self.is_inferencing = True
        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.infer_thread.start()

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        if not self.is_capturing and not self.is_inferencing:
            return

        self.is_capturing = False
        self.is_inferencing = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=2.0)

        if self.ia:
            try:
                self.ia.stop()
                self.ia.destroy()
            except Exception:
                pass
            self.ia = None

        if self.h:
            try:
                self.h.reset()
            except Exception:
                pass
            self.h = None

        if self.ctx:
            try:
                self.ctx.pop()
            except Exception:
                pass
            try:
                self.ctx.detach()
            except Exception:
                pass
            self.ctx = None

    def _capture_loop(self):
        try:
            self.ia.start()
            frame_count = 0
            while self.is_capturing:
                try:
                    with self.ia.fetch(timeout=2000) as buffer:
                        frame_rgb = self._process_frame(buffer)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        if not self.display_q.full():
                            self.display_q.put(frame_bgr)

                        h, w = frame_bgr.shape[:2]
                        scale = min(640 / h, 640 / w)
                        nw, nh = int(round(w * scale)), int(round(h * scale))
                        resized = cv2.resize(frame_bgr, (nw, nh))
                        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
                        top, left = (640 - nh) // 2, (640 - nw) // 2
                        canvas[top:top + nh, left:left + nw] = resized

                        try:
                            self.infer_q.put_nowait((canvas, time.perf_counter()))
                        except queue.Full:
                            pass

                        frame_count += 1

                except Exception as e:
                    if self.is_capturing:
                        time.sleep(0.01)

        except Exception as e:
            pass

    def _inference_loop(self):
        if self.detector is None or self.ctx is None:
            return

        self.ctx.push()

        done = deque()
        last_cap_ts = None
        try:
            while self.is_inferencing:
                try:
                    canvas, cap_ts = self.infer_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                cam_fps = 0.0
                if last_cap_ts is not None:
                    dt = max(1e-9, (cap_ts - last_cap_ts))
                    cam_fps = 1.0 / dt
                last_cap_ts = cap_ts

                t0 = time.perf_counter()
                detections = self.detector.detect_raw_frame(canvas, score_threshold=self.conf_threshold)
                infer_ms = (time.perf_counter() - t0) * 1e3

                t1 = time.perf_counter()
                vis = self.visualize_detections_img(canvas, detections)
                post_ms = (time.perf_counter() - t1) * 1e3

                now = time.perf_counter()
                done.append(now)
                while done and (now - done[0] > 1.0):
                    done.popleft()
                disp_fps = float(len(done))

                det_json = [
                    {"bbox": det.bbox, "label": int(det.label_id), "score": float(det.score)}
                    for det in detections
                ]

                ok, jpeg = cv2.imencode(".jpg", vis)
                if not ok:
                    continue
                b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

                payload = {
                    "frame": b64,
                    "detections": det_json,
                    "metrics": {
                        "camera_fps": cam_fps,
                        "display_fps": disp_fps,
                        "infer_ms": infer_ms,
                        "post_ms": post_ms,
                        "conf_threshold": self.conf_threshold
                    }
                }

                if self.socketio:
                    self.socketio.emit("inference_result", payload, to="stream", namespace="/ws")
                self.infer_q.task_done()

        finally:
            self.ctx.pop()

    def _process_frame(self, buffer) -> np.ndarray:
        c = buffer.payload.components[0]
        w, h, data = c.width, c.height, c.data

        if self.pixel_format == "RGB8":
            img = data.reshape((h, w, 3))
            if self.vendor and self.vendor.lower().startswith("allied vision"):
                img = img[..., ::-1]
        elif self.pixel_format == "BGR8":
            bgr = data.reshape((h, w, 3))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif self.pixel_format == "Mono8":
            mono = data.reshape((h, w))
            img = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
        elif self.pixel_format and self.pixel_format.startswith("Bayer"):
            raw = data.reshape((h, w))
            if "RG" in self.pixel_format:
                img = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
            elif "GR" in self.pixel_format:
                img = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)
            elif "BG" in self.pixel_format:
                img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB)
            elif "GB" in self.pixel_format:
                img = cv2.cvtColor(raw, cv2.COLOR_BayerGB2RGB)
            else:
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        else:
            gray = data.reshape((h, w))
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if not img.flags.writeable:
            img = img.copy()
        return img

    def visualize_detections_img(self, img, detections):
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{det.label_id}: {det.score:.2f}"
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def client_joined(self):
        with self.clients_lock:
            self.client_count += 1
            if self.client_count == 1:
                self.start()
        return self.client_count

    def client_left(self):
        with self.clients_lock:
            self.client_count = max(0, self.client_count - 1)
            if self.client_count == 0:
                self.stop()
        return self.client_count

    def set_confidence(self, value: float):
        self.conf_threshold = float(max(0.05, min(0.99, value)))
        return self.conf_threshold

    def run_with_inference(self, engine_path):
        self.set_engine_path(engine_path)
        self._init_cuda()
        self._init_camera()
        self._init_tensorrt()
        self.start()

    def run_normal(self):
        config = self.app_config.get_config()
        cti_path = config.get('cti_file_location')
        self._init_cuda()
        self._init_camera()
        self.start()