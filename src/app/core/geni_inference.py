import time
import torch
import threading
from harvesters.core import Harvester
import numpy as np
import cv2
import queue
import json
import tensorrt as trt
import pycuda.driver as cuda
import mmcv
import sys
import os
from typing import Dict, List, Optional
import signal
from collections import deque
import base64
import gc
import threading

try:
    import tkinter as tk
    from tkinter import filedialog

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

cuda.init()


class DetectionResult:
    def __init__(
        self,
        bbox: List[float],
        label_id: int,
        score: float,
        mask: Optional[np.ndarray] = None,
    ):
        self.bbox = bbox
        self.label_id = label_id
        self.score = score
        self.mask = mask


class Transform:
    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, data: Dict):
        return data


class LoadImageFromFile(Transform):
    def __call__(self, data: Dict) -> Dict:
        if "img_path" in data and not "img" in data:
            img_path = data["img_path"]
            img = mmcv.imread(img_path, flag="color")
            data["img"] = img
            data["ori_shape"] = img.shape[:2]
            data["img_shape"] = img.shape[:2]
            data["img_fields"] = ["img"]
        return data


class Resize(Transform):
    def __call__(self, data: Dict) -> Dict:
        img = data["img"]
        if isinstance(self.params["size"], list):
            width, height = self.params["size"][1], self.params["size"][0]
        else:
            width, height = self.params["size"], self.params["size"]
        keep_ratio = self.params.get("keep_ratio", False)
        h, w = img.shape[:2]
        if keep_ratio:
            scale = min(height / h, width / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_img = mmcv.imresize(img, (new_w, new_h))
            data["scale_factor"] = (new_w / w, new_h / h)
            data["resize_shape"] = (new_h, new_w)
            data["img"] = resized_img
            data["img_shape"] = resized_img.shape[:2]
        else:
            resized_img = mmcv.imresize(img, (width, height))
            data["scale_factor"] = (width / w, height / h)
            data["img"] = resized_img
            data["img_shape"] = resized_img.shape[:2]
        return data


class Pad(Transform):
    def __call__(self, data: Dict) -> Dict:
        img = data["img"]
        h, w = img.shape[:2]
        if "size" in self.params:
            if isinstance(self.params["size"], list):
                pad_h, pad_w = self.params["size"][0], self.params["size"][1]
            else:
                pad_h = pad_w = self.params["size"]
            pad_val = self.params.get("pad_val", {"img": [0, 0, 0]})
            if isinstance(pad_val, dict):
                pad_val = pad_val.get("img", [0, 0, 0])
            pad_bottom = pad_h - h if pad_h > h else 0
            pad_right = pad_w - w if pad_w > w else 0
            pad_colour = (114, 114, 114)
            padded_img = mmcv.impad(img, shape=(pad_h, pad_w), pad_val=pad_colour)
            data["img"] = padded_img
            data["pad_shape"] = padded_img.shape[:2]
            data["pad_param"] = (0, 0, pad_bottom, pad_right)
        elif "size_divisor" in self.params:
            size_divisor = self.params["size_divisor"]
            if size_divisor > 1:
                pad_h = int(np.ceil(h / size_divisor)) * size_divisor - h
                pad_w = int(np.ceil(w / size_divisor)) * size_divisor - w
                if pad_h > 0 or pad_w > 0:
                    pad_val = self.params.get("pad_val", [0, 0, 0])
                    padded_img = cv2.copyMakeBorder(
                        img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val
                    )
                    data["img"] = padded_img
                    data["pad_shape"] = padded_img.shape[:2]
                    data["pad_param"] = (0, 0, pad_h, pad_w)
        return data


class Normalize(Transform):
    def __call__(self, data: Dict) -> Dict:
        img = data["img"]
        mean = np.array(self.params.get("mean", [0, 0, 0]), dtype=np.float32)
        std = np.array(self.params.get("std", [1, 1, 1]), dtype=np.float32)
        to_rgb = self.params.get("to_rgb", False)
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32, order="C")
        img = (img - mean) / std
        data["img"] = img
        data["img_norm_cfg"] = dict(mean=mean, std=std, to_rgb=to_rgb)
        return data


class DefaultFormatBundle(Transform):
    def __call__(self, data: Dict) -> Dict:
        img = data["img"]
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        img = img.transpose(2, 0, 1)
        data["img"] = img
        return data


class Collect(Transform):
    def __call__(self, data: Dict) -> Dict:
        keys = self.params.get("keys", ["img"])
        meta_keys = self.params.get("meta_keys", [])
        collected = {}
        img_meta = {}
        for key in keys:
            if key in data:
                collected[key] = data[key]
        for key in meta_keys:
            if key in data:
                img_meta[key] = data[key]
        if img_meta:
            collected["img_metas"] = img_meta
        return collected


def build_transform(transform_cfg: Dict) -> Transform:
    transform_type = transform_cfg.get("type", "")
    transform_params = {k: v for k, v in transform_cfg.items() if k != "type"}
    transform_map = {
        "LoadImageFromFile": LoadImageFromFile,
        "Resize": Resize,
        "Pad": Pad,
        "Normalize": Normalize,
        "DefaultFormatBundle": DefaultFormatBundle,
        "Collect": Collect,
    }
    if transform_type in transform_map:
        return transform_map[transform_type](**transform_params)
    return Transform(**transform_params)


class Pipeline:
    def __init__(self, pipeline_path: str, input_name: str = "input"):
        self.transforms = []
        self.input_name = input_name
        if os.path.exists(pipeline_path):
            with open(pipeline_path, "r") as f:
                pipeline_cfg = json.load(f)
            if "pipeline" in pipeline_cfg:
                pipeline_tasks = pipeline_cfg["pipeline"]
                for task in pipeline_tasks.get("tasks", []):
                    if (
                        task.get("name") == "Preprocess"
                        and task.get("module") == "Transform"
                    ):
                        transform_cfgs = task.get("transforms", [])
                        for transform_cfg in transform_cfgs:
                            if transform_cfg.get("type") != "LoadImageFromFile":
                                self.transforms.append(build_transform(transform_cfg))
                for task in pipeline_tasks.get("tasks", []):
                    if task.get("name") == "rtmdet" and task.get("module") == "Net":
                        if "input_map" in task and isinstance(task["input_map"], dict):
                            for k, v in task["input_map"].items():
                                if k == "img":
                                    self.input_name = v
        else:
            self.transforms = [
                Resize(size=[640, 640], keep_ratio=True),
                Pad(size=[640, 640], pad_val={"img": [114, 114, 114]}),
                Normalize(
                    mean=[103.53, 116.28, 123.675],
                    std=[57.375, 57.12, 58.395],
                    to_rgb=False,
                ),
                DefaultFormatBundle(),
                Collect(
                    keys=["img"], meta_keys=["ori_shape", "img_shape", "scale_factor"]
                ),
            ]

    def __call__(self, img: np.ndarray, ori_shape=None) -> (Dict, Dict):
        data = {"img": img}
        if ori_shape is not None:
            data["ori_shape"] = ori_shape
        for transform in self.transforms:
            data = transform(data)
        processed_img = None
        img_metas = {}
        if isinstance(data, dict):
            if "img" in data:
                processed_img = data["img"]
            elif "img_metas" in data and "img" in data.get("img_metas", {}):
                processed_img = data["img_metas"]["img"]
            if "img_metas" in data:
                img_metas = data["img_metas"]
            else:
                img_metas = {
                    k: v
                    for k, v in data.items()
                    if k != "img" and not isinstance(v, np.ndarray)
                }
        if processed_img is None:
            raise ValueError("Failed to extract processed image from pipeline output")
        if len(processed_img.shape) == 3:
            processed_img = np.expand_dims(processed_img, axis=0)
        inputs = {self.input_name: processed_img}
        return inputs, data


class TensorRTDetector:
    def __init__(self, model_path: str, max_detections: int = 1000):
        self.model_path = model_path
        self.max_detections = max_detections
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"Loading TensorRT engine from {model_path}")
        with open(model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {model_path}")

        self.context = self.engine.create_execution_context()
        self._init_io_buffers()
        self.orig_height = None
        self.orig_width = None
        self.input_height = None
        self.input_width = None

    def _init_io_buffers(self):
        assert (
            cuda.Context.get_current() is not None
        ), "No active CUDA context! Make sure ctx.push() was called beforehand."

        print("Analyzing engine I/O tensors")
        self.input_names = []
        self.output_names = []
        self.tensor_shapes = {}
        self.tensor_dtypes = {}

        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                if -1 in shape:
                    if mode == trt.TensorIOMode.INPUT:
                        print(f"Input tensor '{name}' has dynamic shape: {shape}")
                    else:
                        fixed_shape = list(shape)
                        for i, dim in enumerate(fixed_shape):
                            if dim == -1:
                                fixed_shape[i] = self.max_detections
                        shape = tuple(fixed_shape)
                self.tensor_shapes[name] = shape
                self.tensor_dtypes[name] = dtype
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
                if is_input:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                if -1 in shape:
                    if is_input:
                        print(f"Input tensor '{name}' has dynamic shape: {shape}")
                    else:
                        fixed_shape = list(shape)
                        for i, dim in enumerate(fixed_shape):
                            if dim == -1:
                                fixed_shape[i] = self.max_detections
                        shape = tuple(fixed_shape)
                self.tensor_shapes[name] = shape
                self.tensor_dtypes[name] = dtype

        if "raw_input" in self.input_names:
            self.tensor_shapes["raw_input"] = (1, 640, 640, 3)
            self.tensor_dtypes["raw_input"] = trt.DataType.UINT8

        self.host_buffers = {}
        self.device_buffers = {}
        for name in self.input_names + self.output_names:
            shape = self.tensor_shapes[name]
            np_dtype = trt.nptype(self.tensor_dtypes[name])
            size = int(np.prod(shape))
            host_buffer = cuda.pagelocked_empty(size, np_dtype)
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            self.host_buffers[name] = host_buffer
            self.device_buffers[name] = device_buffer

    def detect_raw_frame(
        self, frame_uint8: np.ndarray, score_threshold: float = 0.4
    ) -> List[DetectionResult]:
        stream = cuda.Stream()
        t_gpu0 = time.perf_counter()

        np.copyto(self.host_buffers["raw_input"], frame_uint8.ravel())
        cuda.memcpy_htod_async(
            self.device_buffers["raw_input"], self.host_buffers["raw_input"], stream
        )

        bindings = []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            bindings.append(int(self.device_buffers[name]))

        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

        outs = {}
        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            dtype = np.float32 if name == "dets" else np.int32
            elems = int(np.prod(shape))

            if name not in self.host_buffers or self.host_buffers[name].size < elems:
                if name in self.device_buffers:
                    self.device_buffers[name].free()
                self.host_buffers[name] = cuda.pagelocked_empty(elems, dtype=dtype)
                self.device_buffers[name] = cuda.mem_alloc(
                    elems * np.dtype(dtype).itemsize
                )

            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name])
            outs[name] = self.host_buffers[name][:elems].reshape(shape)

        stream.synchronize()
        gpu_ms = (time.perf_counter() - t_gpu0) * 1e3
        # print(f"[GPU ] {gpu_ms:5.1f} ms")

        dets = outs["dets"]
        labels = outs["labels"]
        results = []
        for i in range(dets.shape[1]):
            score = float(dets[0, i, 4])
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = map(float, dets[0, i, :4])
            results.append(DetectionResult([x1, y1, x2, y2], int(labels[0, i]), score))

        if not hasattr(self, "_dbg"):
            print("runtime dets shape:", dets.shape, "max score:", dets[..., 4].max())
            self._dbg = True

        return results


def visualize_detections_img(
    img: np.ndarray, detections: List[DetectionResult]
) -> np.ndarray:
    result = img.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = colors[det.label_id % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label_text = f"ID:{det.label_id} {det.score:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1
        )
        cv2.putText(
            result,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    return result


class ResourceManager:
    """Manages cleanup of resources"""

    def __init__(self):
        self.resources = []
        self.cleanup_callbacks = []

    def register_resource(self, resource, cleanup_func=None):
        """Register a resource for cleanup"""
        self.resources.append(resource)
        if cleanup_func:
            self.cleanup_callbacks.append(cleanup_func)

    def cleanup_all(self):
        """Clean up all registered resources"""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Cleanup callback error: {e}")

        self.resources.clear()
        self.cleanup_callbacks.clear()
        gc.collect()


class TensorRTGenICamDetector:
    def __init__(self, cti_file_path=None, websocket_mode=False, lazy_init=False):
        if not HAS_TKINTER and not websocket_mode:
            raise RuntimeError(
                "tkinter is not available. Please install tkinter or use websocket_mode=True."
            )

        self.websocket_mode = websocket_mode
        self.socketio = None
        self.service_running = False
        self.service_stop_event = threading.Event()
        self.h = None
        self.ia = None
        self.vendor = None
        self.pixel_format = None
        self.original_pixel_format = None
        try:
            self.ctx = cuda.Device(0).make_context()
        except Exception as e:
            print(f"Failed to create CUDA context: {e}")

        self.is_streaming = False
        self.streaming_active = False
        self.inference_active = False  
        self._memory_check_counter = 0
        self.frame_queue = queue.Queue(maxsize=10)
        self.inference_queue = queue.Queue(maxsize=15)

        # TensorRT components
        self.detector = None
        self.inference_thread = None
        self.inference_running = False
        self.latest_detections = None
        self.resource_manager = ResourceManager()
        self.detection_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.client_lock = threading.Lock()
        self.camera_system = None
        self.threads = []
        self.frame_cache = deque(maxlen=10)

        self.client_lock = threading.Lock()

        self.connected_clients = 0

        # Performance tracking
        self.camera_frame_count = 0
        self.camera_start_time = None
        self.ts_window = deque(maxlen=60)
        self.lat_window = deque(maxlen=60)
        self.current_confidence_threshold = 0.6

        # Display settings
        self.display_width = 1280
        self.display_height = 720
        self.inference_window = "TensorRT GenICam Detection Results"

        # WebSocket streaming settings
        self.stream_fps = 30
        self.last_websocket_frame_time = 0
        self.websocket_frame_interval = 1.0 / self.stream_fps

        # Initialize components
        if not lazy_init:
            if cti_file_path:
                self.initialize_camera(cti_file_path)
            self.initialize_tensorrt()
            if not self.websocket_mode:
                self.setup_opencv_windows()

    def set_socketio(self, socketio):
        """Set the SocketIO instance for WebSocket communication"""
        self.socketio = socketio
        self.debug_websocket_status()
        print("SocketIO instance set for WebSocket streaming")

    def client_joined(self):
        """Handle client joining"""
        with self.client_lock:
            self.connected_clients += 1
            print(f"Client joined. Total clients: {self.connected_clients}")
            return self.connected_clients

    def client_left(self):
        """Handle client leaving"""
        with self.client_lock:
            self.connected_clients = max(0, self.connected_clients - 1)
            print(f"Client left. Total clients: {self.connected_clients}")

            # Auto-stop if no clients connected
            if self.connected_clients == 0 and self.streaming_active:
                print("No clients connected, auto-stopping stream in 30 seconds...")
                threading.Timer(30.0, self._auto_stop_if_no_clients).start()

            return self.connected_clients

    def set_app_config(self, app_config):
        """Set application configuration"""
        self.app_config = app_config

    def setup_opencv_windows(self):
        """Setup OpenCV window"""
        if self.websocket_mode:
            print("WebSocket mode - skipping OpenCV window setup")
            return
        cv2.namedWindow(self.inference_window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.inference_window, 700, 50)
        cv2.resizeWindow(self.inference_window, self.display_width, self.display_height)
        print("OpenCV inference window created")

    # New methods for

    def complete_initialization(self, model_path, cti_file_path):
        """Complete initialization with provided paths"""
        try:
            print(
                f"Completing initialization with model: {model_path}, CTI: {cti_file_path}"
            )

            if self.h is None:
                self.initialize_camera(cti_file_path)

            if self.detector is None:
                self.initialize_tensorrt_with_path(model_path)

            print("GenICam service initialization completed successfully")
            return True

        except Exception as e:
            print(f"Failed to complete initialization: {e}")
            return False

    def initialize_tensorrt_with_path(self, model_path):
        """Initialize TensorRT with specific model path"""
        try:
            print(f"Initializing TensorRT with model: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            t0 = time.perf_counter()

            if self.ctx:
                self.ctx.push()

            self.detector = TensorRTDetector(model_path, max_detections=1000)
            load_ms = (time.perf_counter() - t0) * 1e3
            print(f"TensorRT engine loaded in {load_ms:.1f} ms")

            if self.ctx:
                self.ctx.pop()

            return True

        except Exception as e:
            print(f"TensorRT initialization error: {e}")
            return False

    def run_service_loop(self):
        """Non-blocking service loop that can be run in a thread"""
        try:
            print("Starting GenICam service loop...")
            self.service_running = True
            self.service_stop_event.clear()

            if self.h and len(self.h.device_info_list) > 0:
                print(f"Found {len(self.h.device_info_list)} camera(s)")
                self.connect_camera(0)
                self.start_streaming()
            else:
                print("No cameras found")

            print("GenICam service running - waiting for commands...")

            while self.service_running and not self.service_stop_event.is_set():
                self.service_stop_event.wait(timeout=1.0)

            print("GenICam service loop stopped")

        except Exception as e:
            print(f"Error in service loop: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup_resources()

    def stop_service(self):
        """Stop the service loop - non-blocking version"""
        print("Stopping GenICam service...")
        self.service_running = False
        self.service_stop_event.set()

        # Stop streaming in a separate thread to avoid blocking
        if self.is_streaming:
            stop_thread = threading.Thread(
                target=self._stop_streaming_async, daemon=True
            )
            stop_thread.start()

        print("Stop signal sent - cleanup will continue in background")

    def _stop_streaming_async(self):
        """Async version of stop streaming to prevent blocking"""
        try:
            print("Stopping streaming asynchronously...")
            self.is_streaming = False
            self.inference_running = False

            # Wait for threads with shorter timeouts
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=1.0)  # Reduced timeout
                if self.inference_thread.is_alive():
                    print("Warning: Inference thread did not stop in time")

            if hasattr(self, "camera_thread") and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1.0)  # Reduced timeout
                if self.camera_thread.is_alive():
                    print("Warning: Camera thread did not stop in time")

            self._cleanup_camera_safe()
            self._cleanup_cuda_safe()

            print("Streaming stopped successfully")

        except Exception as e:
            print(f"Error during async stop: {e}")

    def _cleanup_camera_safe(self):
        """Safely cleanup camera resources"""
        try:
            if self.ia:
                try:
                    if hasattr(self.ia, "is_streaming") and self.ia.is_streaming():
                        self.ia.stop()
                        time.sleep(0.1)  # Brief pause
                except Exception as e:
                    print(f"Error stopping camera stream: {e}")

                try:
                    self.ia.destroy()
                    print("Camera destroyed")
                except Exception as e:
                    print(f"Error destroying camera: {e}")
                finally:
                    self.ia = None

            if self.h:
                try:
                    self.h.reset()
                    print("Harvester reset")
                except Exception as e:
                    print(f"Error resetting harvester: {e}")

        except Exception as e:
            print(f"Error in camera cleanup: {e}")

    def _cleanup_cuda_safe(self):
        """Safely cleanup CUDA context"""
        try:
            if self.ctx is not None:
                try:
                    # Check if context is current before popping
                    try:
                        current_ctx = cuda.Context.get_current()
                        if current_ctx == self.ctx:
                            self.ctx.pop()
                            print("CUDA context popped")
                    except cuda.LogicError:
                        # Context not current, that's fine
                        pass

                    try:
                        self.ctx.detach()
                        print("CUDA context detached")
                    except cuda.LogicError:
                        # Already detached, that's fine
                        pass

                except Exception as e:
                    print(f"Error cleaning CUDA context: {e}")
                finally:
                    self.ctx = None

        except Exception as e:
            print(f"Error in CUDA cleanup: {e}")

    def initialize_camera(self, cti_file_path=None):
        """Initialize GenICam camera using Harvesters"""
        try:
            if cti_file_path is None:
                if self.websocket_mode:
                    print("No CTI file provided - camera will be initialized later")
                    return
                elif HAS_TKINTER:
                    root = tk.Tk()
                    root.withdraw()
                    cti_file_path = filedialog.askopenfilename(
                        title="Select GenTL Producer (.cti) File",
                        filetypes=[("GenTL Producer", "*.cti"), ("All Files", "*.*")],
                    )
                    root.destroy()
                    if not cti_file_path:
                        raise RuntimeError("No CTI file selected")
                else:
                    raise RuntimeError("No CTI file provided and GUI not available")

            if not os.path.exists(cti_file_path):
                raise RuntimeError(f"CTI file not found: {cti_file_path}")

            print(f"Using CTI file: {cti_file_path}")

            self.h = Harvester()
            self.h.add_file(cti_file_path)
            self.h.update()

            if len(self.h.device_info_list) == 0:
                raise RuntimeError("No cameras found")

            print(f"Found {len(self.h.device_info_list)} camera(s)")
            for i, device in enumerate(self.h.device_info_list):
                print(f"Camera {i}: {device}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")

    def select_camera_interactive(self):
        """Interactive camera selection"""
        cameras = self.list_available_cameras()
        if not cameras:
            return None

        if len(cameras) == 1:
            print(
                f"Only one camera found. Using camera 0: {cameras[0]['vendor']} {cameras[0]['model']}"
            )
            return 0

        while True:
            try:
                choice = input(
                    f"Select camera (0-{len(cameras)-1}, or press Enter for camera 0): "
                ).strip()
                if choice == "":
                    return 0
                camera_index = int(choice)
                if 0 <= camera_index < len(cameras):
                    return camera_index
                else:
                    print(f"Please enter a number between 0 and {len(cameras)-1}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nCamera selection cancelled")
                return None

    def list_available_cameras(self):
        """List all available cameras with details"""
        if len(self.h.device_info_list) == 0:
            print("No cameras found")
            return []

        print("Available cameras:")
        cameras = []
        for i, device in enumerate(self.h.device_info_list):
            vendor = getattr(device, "vendor", "Unknown")
            model = getattr(device, "model", "Unknown")
            serial = getattr(device, "serial_number", "Unknown")
            user_name = getattr(device, "user_defined_name", "Unknown")

            camera_info = {
                "index": i,
                "vendor": vendor,
                "model": model,
                "serial": serial,
                "user_name": user_name,
                "device": device,
            }
            cameras.append(camera_info)

            print(f"  [{i}] {vendor} {model}")
            print(f"      Serial: {serial}")
            if user_name != "Unknown":
                print(f"      User Name: {user_name}")
            print()

        return cameras

    def initialize_tensorrt(self):
        """Initialize TensorRT detector with default path"""
        try:
            # Default path - will be overridden by initialize_tensorrt_with_path
            engine_path = r"E:\Workspace\Eman\Vim X\models\largefp16.engine"

            if not os.path.exists(engine_path):
                print(f"Engine not found at default path: {engine_path}")
                print("TensorRT will be initialized later with specific path")
                return

            return self.initialize_tensorrt_with_path(engine_path)

        except Exception as e:
            print(f"TensorRT initialization error: {e}")
            return False

    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string for WebSocket transmission"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            return jpg_as_text
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return None

    def emit_websocket_frame(self, frame_with_detections, metadata=None):
        """Emit frame through WebSocket - thread-safe version"""
        if not self.socketio or not self.websocket_mode:
            return

        try:
            # Throttle WebSocket frames to target FPS
            current_time = time.time()
            if (
                current_time - self.last_websocket_frame_time
                < self.websocket_frame_interval
            ):
                return

            self.last_websocket_frame_time = current_time

            # Convert frame to base64
            base64_frame = self.frame_to_base64(frame_with_detections)
            if base64_frame is None:
                return

            # Prepare data for WebSocket
            data = {
                "frame": base64_frame,
                "timestamp": current_time,
                "format": "jpeg",
                "metadata": metadata or {},
            }

            # Use SocketIO's thread-safe emission method
            self.socketio.emit("stream_frame", data, namespace="/ws")

        except Exception as e:
            print(f"Error emitting WebSocket frame: {e}")

    def inference_worker(self):
        """TensorRT inference worker thread with WebSocket support"""
        done_timestamps = deque()
        if self.detector is None or self.ctx is None:
            print("TensorRT not initialized.")
            return

        self.ctx.push()
        print("TensorRT inference worker running")

        try:
            last_cap_ts = None
            while self.inference_running:
                try:
                    try:
                        frame, cap_ts = self.inference_queue.get_nowait()
                    except queue.Empty:
                        time.sleep(0.001)  # Small sleep to prevent busy waiting
                        continue

                    # Run TensorRT inference
                    t0 = time.time()
                    detections = self.detector.detect_raw_frame(
                        frame, score_threshold=self.current_confidence_threshold
                    )
                    latency = (time.time() - t0) * 1000

                    # Visualize detections
                    vis = visualize_detections_img(frame, detections)

                    # Calculate display FPS
                    now = time.perf_counter()
                    done_timestamps.append(now)
                    while done_timestamps and now - done_timestamps[0] > 1.0:
                        done_timestamps.popleft()
                    disp_fps = len(done_timestamps)

                    # Add FPS text to visualization
                    cv2.putText(
                        vis,
                        f"{disp_fps:5.0f} DISPLAY FPS",
                        (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    # Add detection count
                    cv2.putText(
                        vis,
                        f"{len(detections)} DETECTIONS",
                        (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # Prepare metadata
                    metadata = {
                        "detection_count": len(detections),
                        "inference_latency_ms": latency,
                        "processing_fps": disp_fps,
                        "confidence_threshold": self.current_confidence_threshold,
                    }

                    # Update latest detections
                    with self.detection_lock:
                        self.latest_detections = vis

                    # Emit frame via WebSocket
                    self.emit_websocket_frame(vis, metadata)

                    self.inference_queue.task_done()

                except Exception as e:
                    print(f"Inference worker error: {e}")
        finally:
            self.ctx.pop()
            print("TensorRT inference worker stopped")

    def debug_websocket_status(self):
        """Debug method to check WebSocket status"""
        print(f"WebSocket Mode: {self.websocket_mode}")
        print(f"SocketIO Instance: {self.socketio is not None}")
        print(f"Streaming Active: {self.is_streaming}")
        print(f"Inference Running: {self.inference_running}")
        if self.socketio:
            print("SocketIO server is available")

    def get_status(self):
        """Get current streaming status for WebSocket clients"""
        return {
            "is_streaming": self.is_streaming,
            "inference_running": self.inference_running,
            "camera_connected": self.ia is not None,
            "confidence_threshold": self.current_confidence_threshold,
            "camera_info": (
                {
                    "vendor": self.vendor if self.vendor else "Unknown",
                    "pixel_format": (
                        self.pixel_format if self.pixel_format else "Unknown"
                    ),
                    "resolution": f"{self.display_width}x{self.display_height}",
                }
                if self.ia
                else None
            ),
        }

    def get_stream_status(self):
        """Get current streaming status for WebSocket clients"""
        return {
            "is_streaming": self.is_streaming,
            "inference_running": self.inference_running,
            "camera_connected": self.ia is not None,
            "confidence_threshold": self.current_confidence_threshold,
            "camera_info": (
                {
                    "vendor": self.vendor if self.vendor else "Unknown",
                    "pixel_format": (
                        self.pixel_format if self.pixel_format else "Unknown"
                    ),
                    "resolution": f"{self.display_width}x{self.display_height}",
                }
                if self.ia
                else None
            ),
        }

    def run_normal(self):
        """Non-blocking version - just prepares the service"""
        try:
            print("Starting GenICam service (non-blocking)")

            if not self.websocket_mode:
                self.websocket_mode = True

            # Just check if cameras are available - don't start anything yet
            if self.h and len(self.h.device_info_list) > 0:
                print(f"Found {len(self.h.device_info_list)} camera(s)")
                return True
            else:
                print("No cameras found")
                return False

        except Exception as e:
            print(f"Error in run_normal: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def stop(self):
        """Stop streaming with comprehensive cleanup"""
        try:
            print("Stopping GenICam service...")

            # Signal shutdown
            self.streaming_active = False
            self.inference_active = False
            self.shutdown_event.set()

            # Stop camera system - use new method if available
            if self.camera_system:
                if hasattr(self.camera_system, "stop_streaming"):
                    self.camera_system.stop_streaming()
                else:
                    # Fallback to old method
                    self.camera_system.stop_streaming()

            # Wait for threads to finish
            self._cleanup_threads()

            # Clear frame queues and cache
            self._clear_frame_queues()
            self._clear_frame_cache()

            # Notify clients
            if self.socketio:
                self.socketio.emit(
                    "status",
                    {"message": "Stream stopped", "streaming_active": False},
                    namespace="/ws",
                    to="stream",
                )

            # Force garbage collection
            gc.collect()

            print("GenICam service stopped successfully")
            return True

        except Exception as e:
            print(f"Failed to stop service: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _cleanup_threads(self):
        """Clean up all threads"""
        for thread in self.threads:
            if thread.is_alive():
                try:
                    thread.join(timeout=3.0)
                    if thread.is_alive():
                        print(f"Warning: Thread {thread.name} did not stop cleanly")
                except Exception as e:
                    print(f"Error stopping thread {thread.name}: {e}")

        self.threads.clear()
        self.websocket_thread = None

    def _clear_frame_queues(self):
        """Clear all frame queues"""
        queues_to_clear = [self.frame_queue]
        if self.camera_system:
            if hasattr(self.camera_system, "frame_queue"):
                queues_to_clear.append(self.camera_system.frame_queue)
            if hasattr(self.camera_system, "inference_queue"):
                queues_to_clear.append(self.camera_system.inference_queue)

        for q in queues_to_clear:
            while not q.empty():
                try:
                    frame = q.get_nowait()
                    if isinstance(frame, (tuple, list)) and len(frame) > 0:
                        del frame[0]
                    del frame
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Error clearing queue: {e}")

    def _clear_frame_cache(self):
        """Clear frame cache"""
        self.frame_cache.clear()
        gc.collect()

    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed"""
        self._memory_check_counter += 1
        if self._memory_check_counter % 50 == 0:  # Check every 50 frames
            if not self.memory_monitor.check_memory():
                self._clear_frame_cache()
                gc.collect()

    def _calculate_display_size(
        self, width, height, max_display_width=1920, max_display_height=1080
    ):
        width_scale = max_display_width / width
        height_scale = max_display_height / height
        scale = min(width_scale, height_scale, 1.0)
        return int(width * scale), int(height * scale)

    # def process_frame(self, buffer):
    #     """Process frame from GenICam camera to RGB format"""
    #     component = buffer.payload.components[0]
    #     width, height, data = component.width, component.height, component.data

    #     # Handle different pixel formats
    #     if self.pixel_format == "RGB8":
    #         image = data.reshape((height, width, 3))
    #         if self.vendor.lower().startswith("allied vision"):
    #             image = image[..., ::-1]  # Swap channels for Allied Vision
    #     elif self.pixel_format == "BGR8":
    #         bgr = data.reshape((height, width, 3))
    #         image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #     elif self.pixel_format == "Mono8":
    #         mono = data.reshape((height, width))
    #         image = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
    #     elif self.pixel_format.startswith("Bayer"):
    #         raw = data.reshape((height, width))
    #         if "RG" in self.pixel_format:
    #             image = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
    #         elif "GR" in self.pixel_format:
    #             image = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)
    #         elif "BG" in self.pixel_format:
    #             image = cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB)
    #         elif "GB" in self.pixel_format:
    #             image = cv2.cvtColor(raw, cv2.COLOR_BayerGB2RGB)
    #         else:
    #             image = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    #     else:
    #         print(f"[Warning] Unknown format {self.pixel_format}, fallback gray")
    #         gray = data.reshape((height, width))
    #         image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    #     if image.dtype != np.uint8:
    #         image = image.astype(np.uint8)
    #     if not image.flags.writeable:
    #         image = image.copy()
    #     return image

    def process_frame(self, buffer):
        """Process frame from GenICam camera to RGB format"""
        component = buffer.payload.components[0]
        width, height, data = component.width, component.height, component.data

        # Handle different pixel formats
        if self.pixel_format == "RGB8":
            image = data.reshape((height, width, 3))
            if self.vendor.lower().startswith("allied vision"):
                image = image[..., ::-1]  # Swap channels for Allied Vision

        elif self.pixel_format == "BGR8":
            bgr = data.reshape((height, width, 3))
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif self.pixel_format == "Mono8":
            mono = data.reshape((height, width))
            image = cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
        elif self.pixel_format.startswith("Bayer"):
            raw = data.reshape((height, width))
            if "RG" in self.pixel_format:
                image = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
            elif "GR" in self.pixel_format:
                image = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)
            elif "BG" in self.pixel_format:
                image = cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB)
            elif "GB" in self.pixel_format:
                image = cv2.cvtColor(raw, cv2.COLOR_BayerGB2RGB)
            else:
                image = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        else:
            print(f"[Warning] Unknown format {self.pixel_format}, fallback gray")
            gray = data.reshape((height, width))
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if not image.flags.writeable:
            image = image.copy()
        return image

    

    def camera_capture_worker(self):
        """Camera capture worker thread"""
        try:
            self.ia.start()
            frame_count = 0

            while self.is_streaming:
                try:
                    with self.ia.fetch(timeout=2000) as buffer:
                        # Process frame to RGB format
                        frame_rgb = self.process_frame(buffer)

                        # Convert RGB to BGR for OpenCV processing
                        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        current_time = time.perf_counter()

                        # Update frame counting
                        frame_count += 1
                        if self.camera_start_time is None:
                            self.camera_start_time = current_time

                        # Send to inference queue with letterboxing
                        if self.inference_running and not self.inference_queue.full():
                            # Letterbox to 640x640
                            h, w = frame_rgb.shape[:2]
                            scale = min(640 / h, 640 / w)
                            nw, nh = int(round(w * scale)), int(round(h * scale))
                            resized = cv2.resize(frame_rgb, (nw, nh))
                            canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
                            top, left = (640 - nh) // 2, (640 - nw) // 2
                            canvas[top : top + nh, left : left + nw] = resized

                            try:
                                self.inference_queue.put_nowait((canvas, current_time))
                            except queue.Full:
                                pass

                        # Send to display queue
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame_bgr)

                        if frame_count % 100 == 0:
                            print(f"Frame {frame_count} processed")

                except Exception as e:
                    if self.is_streaming:
                        print(f"Frame capture error: {e}")
                        time.sleep(0.01)

        except Exception as e:
            print(f"Camera worker error: {e}")

    def update_confidence_threshold(self, threshold):
        """Update confidence threshold via WebSocket command"""
        if 0.1 <= threshold <= 0.9:
            self.current_confidence_threshold = threshold
            return {"success": True, "threshold": threshold}
        return {"success": False, "error": "Threshold must be between 0.1 and 0.9"}

    def websocket_start_streaming(self, camera_index=0):
        """Start streaming specifically for WebSocket mode"""
        print("Streaming")
        if self.is_streaming:
            return {"success": False, "error": "Already streaming"}

        try:
            # Connect to camera if not already connected
            if self.ia is None:
                if camera_index >= len(self.h.device_info_list):
                    return {
                        "success": False,
                        "error": f"Camera index {camera_index} not available",
                    }
                self.connect_camera(camera_index)

            self.start_streaming()
            return {"success": True, "message": "Streaming started"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def websocket_stop_streaming(self):
        """Stop streaming for WebSocket mode"""
        try:
            self.stop_streaming()
            return {"success": True, "message": "Streaming stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_streaming(self):
        """Start camera streaming with TensorRT inference - FIXED VERSION"""
        if self.is_streaming or getattr(self, 'streaming_active', False):
            print("Streaming already active")
            return

        try:
            if self.detector is None:
                print("TensorRT not initialized. Please restart the application.")
                return

            if self.ia is None:
                print("Camera not connected.")
                return

            # Recreate CUDA context if needed
            if self.ctx is None:
                try:
                    self.ctx = cuda.Device(0).make_context()
                    print("CUDA context created for streaming")
                except Exception as e:
                    print(f"Failed to create CUDA context: {e}")
                    return

            print("Starting streaming...")

            # Clear any residual data before starting
            self._clear_all_queues()
            
            # Reset all streaming states
            self.shutdown_event.clear()
            self.service_stop_event.clear()
            
            # Reset counters
            self.camera_frame_count = 0
            self.camera_start_time = None

            # Set streaming flags in correct order
            self.inference_running = True
            self.inference_active = True
            self.is_streaming = True
            self.streaming_active = True

            # Start inference worker
            self.inference_thread = threading.Thread(
                target=self.inference_worker, daemon=True, name="InferenceWorker"
            )
            self.inference_thread.start()

            # Start camera capture worker
            self.camera_thread = threading.Thread(
                target=self.camera_capture_worker, daemon=True, name="CameraWorker"
            )
            self.camera_thread.start()

            print("Streaming started with TensorRT inference...")

        except Exception as e:
            # If start fails, reset states
            self.is_streaming = False
            self.streaming_active = False
            self.inference_running = False
            self.inference_active = False
            print(f"Failed to start streaming: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop_streaming(self):
        """Non-blocking stop streaming - calls complete stop"""
        if not self.is_streaming:
            print("Streaming not active")
            return

        print("Initiating stream stop...")
        return self.stop_streaming_completely()

    def run_websocket_mode(self):
        """Run in WebSocket mode (no OpenCV display)"""
        print("Running in WebSocket mode - no GUI display")

        if len(self.h.device_info_list) > 0:
            self.connect_camera(0)
            print("Camera connected. Ready for WebSocket commands.")
            self.websocket_start_streaming(camera_index=0)
        else:
            print("No cameras found. Waiting for commands...")

        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_streaming()
            print("WebSocket mode cleanup completed")

    def cleanup_resources(self):
        """Cleanup resources - calls complete stop"""
        print("Starting resource cleanup...")
        try:
            self.stop_streaming_completely()
            print("Resource cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


    def cleanup_all(self):
        """Comprehensive cleanup of all resources"""
        print("Starting GenICam service cleanup...")
        try:
            self.stop_streaming_completely()
            
            # Additional cleanup
            if hasattr(self, 'resource_manager') and self.resource_manager:
                self.resource_manager.cleanup_all()
            
            with self.client_lock:
                self.connected_clients = 0

            import gc
            gc.collect()

            print("GenICam service cleanup completed")

        except Exception as e:
            print(f"Error during GenICam cleanup: {e}")


    def _check_memory_usage(self):
        """Simplified memory check without external monitor"""
        import gc
        self._memory_check_counter = getattr(self, '_memory_check_counter', 0) + 1
        
        if self._memory_check_counter % 50 == 0:  # Check every 50 frames
            # Simple cleanup every so often
            collected = gc.collect()
            if collected > 0:
                print(f"Garbage collected {collected} objects")
            
            # Clear frame cache if it gets too large
            if hasattr(self, 'frame_cache') and len(self.frame_cache) > 5:
                self.frame_cache.clear()

    def _async_cleanup(self):
        try:
            time.sleep(0.5)

            self._cleanup_camera_safe()
            self._cleanup_cuda_safe()
            self._clear_queues_safe()

            gc.collect()
            print("Async cleanup completed")

        except Exception as e:
            print(f"Error in async cleanup: {e}")

    def _clear_queues_safe(self):
        """Safely clear all queues without blocking"""
        try:

            for queue_obj in [self.frame_queue, self.inference_queue]:
                cleared = 0
                while not queue_obj.empty() and cleared < 100:
                    try:
                        queue_obj.get_nowait()
                        cleared += 1
                    except:
                        break

            if hasattr(self, "frame_cache"):
                self.frame_cache.clear()

            print(f"Queues cleared")

        except Exception as e:
            print(f"Error clearing queues: {e}")

    def run(self):
        """Main run loop - chooses between GUI and WebSocket mode"""
        if self.websocket_mode:
            self.run_websocket_mode()
        else:
            self.run_gui_mode()

    def display_frames(self):
        """Display frames with inference results"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                with self.detection_lock:
                    frame_with_detections = (
                        self.latest_detections
                        if self.latest_detections is not None
                        else frame
                    )

                # Convert BGR to RGB for proper color display
                frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                cv2.imshow(self.inference_window, frame_rgb)
        except Exception as e:
            print(f"Display error: {e}")

    def _force_camera_release(self):
        """Force release of camera resources"""
        try:
            print("Force releasing camera resources...")
            
            # Try to create and immediately destroy any existing connections
            if self.h and len(self.h.device_info_list) > 0:
                for i in range(len(self.h.device_info_list)):
                    try:
                        temp_ia = self.h.create(i)
                        if temp_ia:
                            temp_ia.destroy()
                        print(f"Force released camera {i}")
                    except Exception as e:
                        # Expected to fail if camera is locked
                        pass
            
            # Force a small delay to let resources settle
            time.sleep(1.0)
            
        except Exception as e:
            print(f"Error in force camera release: {e}")


    def _refresh_camera_list(self):
        """Refresh the camera device list"""
        try:
            print("Refreshing camera device list...")
            if self.h:
                # Update the device list
                self.h.update()
                print(f"Camera list refreshed. Found {len(self.h.device_info_list)} cameras")
                return True
        except Exception as e:
            print(f"Error refreshing camera list: {e}")
            return False


    def connect_camera(self, camera_index=None):
        """Connect to specific camera with resource lock handling - UPDATED"""
        try:
            # Clean up any existing connection more thoroughly
            if self.ia is not None:
                print("Disconnecting existing camera...")
                try:
                    if hasattr(self.ia, 'is_streaming') and self.ia.is_streaming():
                        self.ia.stop()
                        time.sleep(0.3)
                    self.ia.destroy()
                    time.sleep(0.5)  # Longer wait for resource release
                    print("Previous camera disconnected")
                except Exception as e:
                    print(f"Error during camera disconnect: {e}")
                finally:
                    self.ia = None

            # Force release any locked resources
            self._force_camera_release()
            
            # Refresh camera list
            self._refresh_camera_list()

            if camera_index is None:
                if self.websocket_mode:
                    camera_index = 0
                else:
                    camera_index = self.select_camera_interactive()
                    if camera_index is None:
                        raise ValueError("No camera selected")

            if camera_index >= len(self.h.device_info_list):
                raise ValueError(
                    f"Camera index {camera_index} not available. Found {len(self.h.device_info_list)} cameras."
                )

            print(f"Connecting to camera {camera_index}...")
            
            # Enhanced retry mechanism for camera connection with resource lock handling
            max_retries = 5  # Increased retries
            retry_delays = [0.5, 1.0, 2.0, 3.0, 5.0]  # Progressive delays
            
            for attempt in range(max_retries):
                try:
                    print(f"Connection attempt {attempt + 1}/{max_retries}")
                    
                    # Try to create camera connection
                    self.ia = self.h.create(camera_index)
                    if self.ia:
                        print(f"Camera {camera_index} connected successfully")
                        break
                    else:
                        if attempt < max_retries - 1:
                            delay = retry_delays[attempt]
                            print(f"Camera creation failed, waiting {delay}s before retry...")
                            time.sleep(delay)
                            
                            # Try force release again before next attempt
                            if attempt >= 2:  # After 2 failed attempts
                                self._force_camera_release()
                                self._refresh_camera_list()
                                
                except Exception as e:
                    error_msg = str(e)
                    print(f"Camera connection attempt {attempt + 1} failed: {error_msg}")
                    
                    if "already in use" in error_msg.lower() or "resource in use" in error_msg.lower():
                        if attempt < max_retries - 1:
                            delay = retry_delays[attempt]
                            print(f"Camera resource locked. Waiting {delay}s before retry...")
                            time.sleep(delay)
                            
                            # Force release and refresh after resource lock error
                            self._force_camera_release()
                            self._refresh_camera_list()
                        else:
                            # Last attempt failed due to resource lock
                            raise RuntimeError(
                                f"Camera {camera_index} is locked by another process. "
                                f"Please close any other applications using the camera or restart the camera."
                            )
                    else:
                        # Different error - retry with shorter delay
                        if attempt < max_retries - 1:
                            print(f"Retrying in 0.5s...")
                            time.sleep(0.5)
                        else:
                            raise
            
            if not self.ia:
                raise RuntimeError(f"Failed to create camera instance for index {camera_index} after {max_retries} attempts")

            # Configure camera
            node_map = self.ia.remote_device.node_map
            self.vendor = self.h.device_info_list[camera_index].vendor

            self.original_pixel_format = node_map.PixelFormat.value
            print(f"Original pixel format: {self.original_pixel_format}")

            # Get maximum resolution
            max_width = node_map.Width.max
            max_height = node_map.Height.max
            node_map.Width.value = max_width
            node_map.Height.value = max_height

            # Set best available format
            available_formats = list(node_map.PixelFormat.symbolics)
            format_priority = [
                "RGB8", "BGR8", "BayerRG8", "BayerGR8", "BayerBG8", "BayerGB8", "Mono8"
            ]
            selected_format = None
            for fmt in format_priority:
                if fmt in available_formats:
                    try:
                        node_map.PixelFormat.value = fmt
                        selected_format = fmt
                        break
                    except:
                        continue
            
            if not selected_format:
                selected_format = node_map.PixelFormat.value
            self.pixel_format = selected_format

            final_width = node_map.Width.value
            final_height = node_map.Height.value

            print(f"Connected to {self.vendor} camera")
            print(f"Resolution: {final_width}x{final_height}, Format: {self.pixel_format}")

            # Calculate display size
            self.display_width, self.display_height = self._calculate_display_size(
                final_width, final_height
            )
            print(f"Display size: {self.display_width}x{self.display_height}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            if self.ia:
                try:
                    self.ia.destroy()
                except:
                    pass
                self.ia = None
            raise


    def _cleanup_camera_completely(self):
        """Complete camera cleanup - ENHANCED for resource lock prevention"""
        try:
            if self.ia:
                try:
                    print("Destroying camera interface...")
                    # Make sure camera is stopped
                    if hasattr(self.ia, 'is_streaming') and self.ia.is_streaming():
                        self.ia.stop()
                        time.sleep(0.2)
                    
                    # Destroy the interface
                    self.ia.destroy()
                    print("Camera interface destroyed")
                    
                    # Extra wait to ensure resource is fully released
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error destroying camera interface: {e}")
                finally:
                    self.ia = None

            # Force release any remaining resources
            self._force_camera_release()

            # Keep harvester alive for reuse - don't reset it
            if self.h:
                try:
                    # Refresh the device list to clear any stale connections
                    self.h.update()
                    print(f"Harvester available with {len(self.h.device_info_list)} cameras")
                except Exception as e:
                    print(f"Error checking harvester status: {e}")

            # Reset camera-related state
            self.vendor = None
            self.pixel_format = None
            self.original_pixel_format = None

        except Exception as e:
            print(f"Error in camera cleanup: {e}")


    def stop_streaming_completely(self):
        """Completely stop streaming and clean up all resources - ENHANCED VERSION"""
        try:
            print("Initiating complete stream stop...")

            # Set ALL streaming flags to False
            self.is_streaming = False
            self.streaming_active = False
            self.inference_running = False  
            self.inference_active = False
            self.service_running = False
            self.service_stop_event.set()
            self.shutdown_event.set()

            # Stop camera properly BEFORE thread cleanup
            if self.ia:
                try:
                    print("Stopping camera acquisition...")
                    if hasattr(self.ia, 'is_streaming') and self.ia.is_streaming():
                        self.ia.stop()
                        time.sleep(0.3)  # Increased wait time
                        print("Camera acquisition stopped")
                except Exception as e:
                    print(f"Error stopping camera: {e}")

            # Wait for threads with proper timeout and cleanup
            threads_to_cleanup = []
            
            if hasattr(self, 'inference_thread') and self.inference_thread and self.inference_thread.is_alive():
                threads_to_cleanup.append(('inference_thread', self.inference_thread))
                
            if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
                threads_to_cleanup.append(('camera_thread', self.camera_thread))

            for thread_name, thread in threads_to_cleanup:
                try:
                    print(f"Waiting for {thread_name} to stop...")
                    thread.join(timeout=3.0)  # Increased timeout for proper shutdown
                    if thread.is_alive():
                        print(f"Warning: {thread_name} did not stop cleanly within timeout")
                    else:
                        print(f"{thread_name} stopped successfully")
                except Exception as e:
                    print(f"Error stopping {thread_name}: {e}")

            # Clear all queues properly
            self._clear_all_queues()

            # Cleanup camera resources completely with enhanced resource release
            self._cleanup_camera_completely()

            # Cleanup CUDA context properly  
            self._cleanup_cuda_completely()

            # Reset all state variables
            self._reset_all_state()

            # Force garbage collection
            import gc
            gc.collect()

            print("Streaming completely stopped and resources cleaned up")
            return True

        except Exception as e:
            print(f"Error during complete stop: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _clear_all_queues(self):
        """Clear all queues completely - FIXED for tuple deletion error"""
        try:
            queues_to_clear = [
                ('frame_queue', self.frame_queue),
                ('inference_queue', self.inference_queue)
            ]
            
            # Add camera system queues if they exist
            if self.camera_system:
                if hasattr(self.camera_system, 'frame_queue'):
                    queues_to_clear.append(('camera_system.frame_queue', self.camera_system.frame_queue))
                if hasattr(self.camera_system, 'inference_queue'):
                    queues_to_clear.append(('camera_system.inference_queue', self.camera_system.inference_queue))

            for queue_name, queue_obj in queues_to_clear:
                cleared_count = 0
                while not queue_obj.empty() and cleared_count < 100:  # Prevent infinite loop
                    try:
                        item = queue_obj.get_nowait()
                        # Don't try to delete individual tuple items - just delete the whole item
                        del item
                        cleared_count += 1
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Error clearing item from {queue_name}: {e}")
                        break
                
                if cleared_count > 0:
                    print(f"Cleared {cleared_count} items from {queue_name}")

            # Clear frame cache
            if hasattr(self, 'frame_cache'):
                self.frame_cache.clear()
                print("Frame cache cleared")

        except Exception as e:
            print(f"Error clearing queues: {e}")

    

    def _cleanup_cuda_completely(self):
        """Complete CUDA cleanup - FIXED to prevent context stack errors"""
        try:
            if self.ctx is not None:
                try:
                    # Pop context if it's current
                    try:
                        current_ctx = cuda.Context.get_current()
                        if current_ctx == self.ctx:
                            self.ctx.pop()
                            print("CUDA context popped")
                    except cuda.LogicError:
                        # Context not current, that's fine
                        pass

                    # Don't detach - just set to None
                    print("CUDA context cleared")

                except Exception as e:
                    print(f"Error cleaning CUDA context: {e}")
                finally:
                    self.ctx = None

            # Don't recreate context here - let it be created when needed

        except Exception as e:
            print(f"Error in CUDA cleanup: {e}")

    def _reset_all_state(self):
        """Reset all state variables for clean restart"""
        try:
            # Reset streaming states
            self.is_streaming = False
            self.streaming_active = False
            self.inference_running = False
            self.inference_active = False
            self.service_running = False
            
            # Clear events
            self.service_stop_event.clear()
            self.shutdown_event.clear()
            
            # Reset threads
            self.inference_thread = None
            self.camera_thread = None
            
            # Reset detection data
            self.latest_detections = None
            
            # Reset performance counters
            self.camera_frame_count = 0
            self.camera_start_time = None
            if hasattr(self, 'ts_window'):
                self.ts_window.clear()
            if hasattr(self, 'lat_window'):
                self.lat_window.clear()
            
            # Clear thread list
            if hasattr(self, 'threads'):
                self.threads.clear()
            
            print("All state variables reset")
            
        except Exception as e:
            print(f"Error resetting state: {e}")    

    def run_gui_mode(self):
        """Original GUI mode with OpenCV display"""
        print("Starting TensorRT GenICam Detection System (GUI Mode)")
        print("Press 'q' to quit, 's' to start/stop streaming")
        print("Press 'c' to decrease confidence, 'v' to increase confidence")

        # Connect to camera with user selection
        if len(self.h.device_info_list) > 0:
            self.connect_camera()  # Will prompt user to select camera
            self.start_streaming()

        try:
            while True:
                self.display_frames()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quitting...")
                    break
                elif key == ord("s"):
                    if self.is_streaming:
                        print("Stopping streaming...")
                        self.stop_streaming()
                    else:
                        print("Starting streaming...")
                        self.start_streaming()
                elif key == ord("c"):
                    self.current_confidence_threshold = max(
                        0.1, self.current_confidence_threshold - 0.1
                    )
                    print(
                        f"Confidence threshold: {self.current_confidence_threshold:.1f}"
                    )
                elif key == ord("v"):
                    self.current_confidence_threshold = min(
                        0.9, self.current_confidence_threshold + 0.1
                    )
                    print(
                        f"Confidence threshold: {self.current_confidence_threshold:.1f}"
                    )

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_streaming()
            cv2.destroyAllWindows()
            print("GUI mode cleanup completed")


def _sigint_handler(sig, frame):

    print("Ctrl-C received, shutting down...")
    global app
    try:
        app.stop_streaming()
    except Exception:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)


def main():
    """Main function"""
    print("Starting TensorRT GenICam Detection System")
    global app
    app = TensorRTGenICamDetector()
    app.run()


if __name__ == "__main__":
    main()
