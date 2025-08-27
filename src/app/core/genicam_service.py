import os
import time
import base64
import json
import queue
import threading
from collections import deque

import numpy as np
import cv2

from harvesters.core import Harvester
import pycuda.driver as cuda

from app.core.geni_inference import TensorRTDetector, Detection

class GenICamService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenICamService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.app_config = None
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
        self.normal_thread = None
        self.is_capturing = False
        self.is_inferencing = False
        self.is_normal_streaming = False
        self.done_timestamps = deque(maxlen=60)
        self.conf_threshold = 0.6
        self.client_count = 0
        self.clients_lock = threading.Lock()
        self.display_width = 1280
        self.display_height = 720
        self.stream_mode = "none"  

    def set_socketio(self, socketio):
        self.socketio = socketio
        print("SocketIO set on GenICamService")

    def set_engine_path(self, engine_path):
        self.engine_path = engine_path

    def set_app_config(self, app_config):
        self.app_config = app_config
        print("AppConfig set on GenICamService")

    def _init_cuda(self):
        print("Initializing CUDA...")
        try:
            cuda.init()
            self.ctx = cuda.Device(0).make_context()
            print("CUDA initialized successfully")
        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            raise

    def _init_camera(self):
        print("Initializing camera...")
        if not self.app_config:
            raise RuntimeError("AppConfig not set - call set_app_config() first")
            
        config = self.app_config.get_config()
        cti_path = config.get('cti_file_location')
        
        if not cti_path:
            raise RuntimeError("CTI file location not configured")
        
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
        print(f"Camera initialized - Pixel format: {self.pixel_format}")

    def _init_tensorrt(self):
        print(f"Initializing TensorRT with engine: {self.engine_path}")
        if not os.path.exists(self.engine_path):
            raise RuntimeError(f"Engine not found: {self.engine_path}")

        self.detector = TensorRTDetector(self.engine_path, max_detections=1000)
        print("TensorRT initialized successfully")

    def start_inference(self):
        if self.is_capturing or self.is_inferencing:
            print("Already running - stopping first")
            self.stop()
        
        print("Starting inference mode...")
        self.stream_mode = "inference"
        self.is_inferencing = True
        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.infer_thread.start()

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def start_normal(self):
        if self.is_capturing or self.is_normal_streaming:
            print("Already running - stopping first")
            self.stop()
            
        print("Starting normal streaming mode...")
        self.stream_mode = "normal"
        self.is_normal_streaming = True
        self.normal_thread = threading.Thread(target=self._normal_stream_loop, daemon=True)
        self.normal_thread.start()

    def stop(self):
        print("Stopping GenICam service...")
        self.stream_mode = "none"
        self.is_capturing = False
        self.is_inferencing = False
        self.is_normal_streaming = False

        for thread in [self.capture_thread, self.infer_thread, self.normal_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

        if self.ia:
            try:
                self.ia.stop()
                self.ia.destroy()
            except Exception as e:
                print(f"Error stopping camera: {e}")
            self.ia = None

        if self.h:
            try:
                self.h.reset()
            except Exception as e:
                print(f"Error resetting harvester: {e}")
            self.h = None

        if self.ctx:
            try:
                self.ctx.pop()
                self.ctx.detach()
            except Exception as e:
                print(f"Error cleaning up CUDA: {e}")
            self.ctx = None
        
        print("GenICam service stopped")

    def _normal_stream_loop(self):
        """Stream raw camera frames without inference"""
        print("Starting normal stream loop...")
        try:
            self.ia.start()
            frame_count = 0
            
            while self.is_normal_streaming:
                try:
                    with self.ia.fetch(timeout=2000) as buffer:
                        frame_rgb = self._process_frame(buffer)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        
                        h, w = frame_bgr.shape[:2]
                        scale = min(640 / h, 640 / w)
                        nw, nh = int(round(w * scale)), int(round(h * scale))
                        resized = cv2.resize(frame_bgr, (nw, nh))

                        
                        now = time.perf_counter()
                        self.done_timestamps.append(now)
                        while self.done_timestamps and (now - self.done_timestamps[0] > 1.0):
                            self.done_timestamps.popleft()
                        fps = float(len(self.done_timestamps))

                      
                        ok, jpeg = cv2.imencode(".jpg", resized)
                        if not ok:
                            continue
                        b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

                        payload = {
                            "frame": b64,
                            "metrics": {
                                "camera_fps": fps,
                                "display_fps": fps,
                                "conf_threshold": self.conf_threshold
                            }
                        }

                    
                        if self.socketio and self.client_count > 0:
                            try:
                                self.socketio.emit("stream_frame", payload, to="stream", namespace="/ws")
                                if frame_count % 30 == 0:  
                                    print(f"Emitted normal frame {frame_count} to {self.client_count} clients")
                            except Exception as e:
                                print(f"Error emitting normal frame: {e}")

                        frame_count += 1

                except Exception as e:
                    if self.is_normal_streaming:
                        print(f"Normal stream frame error: {e}")
                        time.sleep(0.01)

        except Exception as e:
            print(f"Normal stream loop error: {e}")

    def _capture_loop(self):
        print("Starting capture loop for inference...")
        try:
            self.ia.start()
            while self.is_capturing:
                try:
                    with self.ia.fetch(timeout=2000) as buffer:
                        frame_rgb = self._process_frame(buffer)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                       
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

                except Exception as e:
                    if self.is_capturing:
                        print(f"Capture error: {e}")
                        time.sleep(0.01)

        except Exception as e:
            print(f"Capture loop error: {e}")

    def _inference_loop(self):
        print("Starting inference loop...")
        if self.detector is None or self.ctx is None:
            print("ERROR: Detector or CUDA context not initialized")
            return

        self.ctx.push()
        done = deque()
        last_cap_ts = None
        frame_count = 0
        
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

               
                infer_ms = 0
                detections = []
                if self.detector:
                    t0 = time.perf_counter()
                    detections = self.detector.detect_raw_frame(canvas, score_threshold=self.conf_threshold)
                    infer_ms = (time.perf_counter() - t0) * 1e3

                
                vis = self.visualize_detections_img(canvas, detections)

                
                now = time.perf_counter()
                done.append(now)
                while done and (now - done[0] > 1.0):
                    done.popleft()
                disp_fps = float(len(done))

             
                ok, jpeg = cv2.imencode(".jpg", vis)
                if not ok:
                    continue
                b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

             
                det_json = [
                    {"bbox": det.bbox, "label": int(det.label_id), "score": float(det.score)}
                    for det in detections
                ]

                payload = {
                    "frame": b64,
                    "detections": det_json,
                    "metrics": {
                        "camera_fps": cam_fps,
                        "display_fps": disp_fps,
                        "infer_ms": infer_ms,
                        "conf_threshold": self.conf_threshold
                    }
                }

               
                if self.socketio and self.client_count > 0:
                    try:
                        self.socketio.emit("inference_result", payload, to="stream", namespace="/ws")
                        if frame_count % 30 == 0: 
                            print(f"Emitted inference frame {frame_count} - {len(detections)} detections")
                    except Exception as e:
                        print(f"Error emitting inference frame: {e}")

                frame_count += 1
                self.infer_q.task_done()

        except Exception as e:
            print(f"Inference loop error: {e}")
        finally:
            try:
                self.ctx.pop()
            except Exception as e:
                print(f"Error popping CUDA context: {e}")

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
            print(f"Client joined - total: {self.client_count}")
         
        return self.client_count

    def client_left(self):
        with self.clients_lock:
            self.client_count = max(0, self.client_count - 1)
            print(f"Client left - total: {self.client_count}")
            if self.client_count == 0:
                print("No clients left - stopping stream")
                self.stop()
        return self.client_count

    def set_confidence(self, value: float):
        self.conf_threshold = float(max(0.05, min(0.99, value)))
        print(f"Confidence threshold set to: {self.conf_threshold}")
        return self.conf_threshold

    def run_with_inference(self, engine_path):
        print(f"Starting inference stream with engine: {engine_path}")
        try:
            self.set_engine_path(engine_path)
            self._init_cuda()
            self._init_camera()
            self._init_tensorrt()
            self.start_inference()
        except Exception as e:
            print(f"Error starting inference: {e}")
            raise

    def run_normal(self):
        print("Starting normal stream")
        try:
            self._init_cuda()
            self._init_camera()
            self.start_normal()
        except Exception as e:
            print(f"Error starting normal stream: {e}")
            raise