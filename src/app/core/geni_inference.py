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

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

cuda.init()

# TensorRT inference classes (from your original script)
class DetectionResult:
    def __init__(self, bbox: List[float], label_id: int, score: float, mask: Optional[np.ndarray] = None):
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
            img = mmcv.imread(img_path, flag='color') 
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
                    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_val)
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
        img = np.asarray(img, dtype=np.float32, order='C')
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
        "Collect": Collect
    }
    if transform_type in transform_map:
        return transform_map[transform_type](**transform_params)
    return Transform(**transform_params)

class Pipeline:
    def __init__(self, pipeline_path: str, input_name: str = "input"):
        self.transforms = []
        self.input_name = input_name
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'r') as f:
                pipeline_cfg = json.load(f)
            if "pipeline" in pipeline_cfg:
                pipeline_tasks = pipeline_cfg["pipeline"]
                for task in pipeline_tasks.get("tasks", []):
                    if task.get("name") == "Preprocess" and task.get("module") == "Transform":
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
                Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False),
                DefaultFormatBundle(),
                Collect(keys=["img"], meta_keys=["ori_shape", "img_shape", "scale_factor"])
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
                img_metas = {k: v for k, v in data.items() if k != "img" and not isinstance(v, np.ndarray)}
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
        assert cuda.Context.get_current() is not None, \
            "No active CUDA context! Make sure ctx.push() was called beforehand."

        print("Analyzing engine I/O tensors")
        self.input_names = []
        self.output_names = []
        self.tensor_shapes = {}
        self.tensor_dtypes = {}
        
        if hasattr(self.engine, 'num_io_tensors'):
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

    def detect_raw_frame(self, frame_uint8: np.ndarray, score_threshold: float = 0.4) -> List[DetectionResult]:
        stream = cuda.Stream()
        t_gpu0 = time.perf_counter()   
     
        np.copyto(self.host_buffers["raw_input"], frame_uint8.ravel())
        cuda.memcpy_htod_async(self.device_buffers["raw_input"], self.host_buffers["raw_input"], stream)
     
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
                self.device_buffers[name] = cuda.mem_alloc(elems * np.dtype(dtype).itemsize)
     
            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name])
            outs[name] = self.host_buffers[name][:elems].reshape(shape)
     
        stream.synchronize()
        gpu_ms = (time.perf_counter() - t_gpu0) * 1e3
        print(f"[GPU ] {gpu_ms:5.1f} ms") 

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

def visualize_detections_img(img: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    result = img.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255)
    ]
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = colors[det.label_id % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label_text = f"ID:{det.label_id} {det.score:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(result, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return result

class TensorRTGenICamDetector:
    def __init__(self, cti_file_path=None, websocket_mode=False):
        if not HAS_TKINTER and not websocket_mode:
            raise RuntimeError("tkinter is not available. Please install tkinter or use websocket_mode=True.")
        
        # WebSocket mode flag
        self.websocket_mode = websocket_mode
        self.socketio = None
        
        # Camera variables
        self.h = None
        self.ia = None
        self.vendor = None
        self.pixel_format = None
        self.original_pixel_format = None
        self.ctx = cuda.Device(0).make_context()
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.inference_queue = queue.Queue(maxsize=15)
        
        # TensorRT components
        self.detector = None
        self.inference_thread = None
        self.inference_running = False
        self.latest_detections = None
        self.detection_lock = threading.Lock()
        
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
        self.stream_fps = 30  # Target FPS for WebSocket streaming
        self.last_websocket_frame_time = 0
        self.websocket_frame_interval = 1.0 / self.stream_fps
        
        # Initialize components
        self.initialize_camera(cti_file_path)
        self.initialize_tensorrt()
        if not self.websocket_mode:
            self.setup_opencv_windows()

    def set_socketio(self, socketio):
        """Set the SocketIO instance for WebSocket communication"""
        self.socketio = socketio
        print("SocketIO instance set for WebSocket streaming")

    def set_app_config(self, app_config):
        """Set application configuration"""
        self.app_config = app_config

    def initialize_camera(self, cti_file_path=None):
        """Initialize GenICam camera using Harvesters"""
        try:
            # Don't use GUI dialogs in web server mode
            if self.websocket_mode and cti_file_path:
                # Use provided CTI file path (from config)
                if not os.path.exists(cti_file_path):
                    raise RuntimeError(f"CTI file not found: {cti_file_path}")
                print(f"Using configured CTI file: {cti_file_path}")
            elif cti_file_path:
                # Use provided CTI file path
                if not os.path.exists(cti_file_path):
                    raise RuntimeError(f"CTI file not found: {cti_file_path}")
                print(f"Using CTI file: {cti_file_path}")
            else:
                # Only use GUI dialog in standalone mode (not websocket mode)
                if not self.websocket_mode and HAS_TKINTER:
                    root = tk.Tk()
                    root.withdraw()
                    
                    cti_file_path = filedialog.askopenfilename(
                        title="Select GenTL Producer (.cti) File",
                        filetypes=[("GenTL Producer", "*.cti"), ("All Files", "*.*")]
                    )
                    
                    root.destroy()
                    
                    if not cti_file_path:
                        raise RuntimeError("No CTI file selected")
                else:
                    raise RuntimeError("No CTI file provided and GUI not available")

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

    # ... [Keep all your existing methods: list_available_cameras, select_camera_interactive, etc.]

    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string for WebSocket transmission"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return jpg_as_text
        except Exception as e:
            print(f"Error converting frame to base64: {e}")
            return None

    def emit_websocket_frame(self, frame_with_detections, metadata=None):
        """Emit frame through WebSocket"""
        if not self.socketio or not self.websocket_mode:
            return
        
        try:
            # Throttle WebSocket frames to target FPS
            current_time = time.time()
            if current_time - self.last_websocket_frame_time < self.websocket_frame_interval:
                return
            
            self.last_websocket_frame_time = current_time
            
            # Convert frame to base64
            base64_frame = self.frame_to_base64(frame_with_detections)
            if base64_frame is None:
                return
            
            # Prepare data for WebSocket
            data = {
                'frame': base64_frame,
                'timestamp': current_time,
                'format': 'jpeg'
            }
            
            # Add metadata if provided
            if metadata:
                data.update(metadata)
            
            # Emit to WebSocket clients
            self.socketio.emit('stream_frame', data, namespace='/ws')
            
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
                        continue

                    if last_cap_ts is not None:
                        cam_fps = 1.0 / (cap_ts - last_cap_ts)
                    else:
                        cam_fps = 0
                    last_cap_ts = cap_ts

                except queue.Empty:
                    continue

                if frame is None:
                    continue

                # Run TensorRT inference
                t0 = time.time()
                detections = self.detector.detect_raw_frame(frame, score_threshold=self.current_confidence_threshold)
                latency = time.time() - t0
                latency = latency * 1000

                # Visualize detections
                t_post0 = time.perf_counter()
                vis = visualize_detections_img(frame, detections)
                post_ms = (time.perf_counter() - t_post0) * 1e3

                # Calculate display FPS
                now = time.perf_counter()
                done_timestamps.append(now)

                while done_timestamps and now - done_timestamps[0] > 1.0:
                    done_timestamps.popleft()

                disp_fps = len(done_timestamps)
                
                # Add FPS text to visualization
                cv2.putText(vis, f"{disp_fps:5.0f} DISPLAY FPS", (10, 78),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Add detection count
                cv2.putText(vis, f"{len(detections)} DETECTIONS", (10, 108),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Prepare metadata for WebSocket
                metadata = {
                    'detection_count': len(detections),
                    'inference_latency_ms': latency,
                    'processing_fps': disp_fps,
                    'confidence_threshold': self.current_confidence_threshold
                }

                # Update latest detections for OpenCV display
                with self.detection_lock:
                    self.latest_detections = vis

                # Emit frame via WebSocket if in websocket mode
                if self.websocket_mode:
                    self.emit_websocket_frame(vis, metadata)

                self.inference_queue.task_done()

        finally:
            self.ctx.pop()
            print("TensorRT inference worker stopped")

    def get_stream_status(self):
        """Get current streaming status for WebSocket clients"""
        return {
            'is_streaming': self.is_streaming,
            'inference_running': self.inference_running,
            'camera_connected': self.ia is not None,
            'confidence_threshold': self.current_confidence_threshold,
            'camera_info': {
                'vendor': self.vendor if self.vendor else 'Unknown',
                'pixel_format': self.pixel_format if self.pixel_format else 'Unknown',
                'resolution': f"{self.display_width}x{self.display_height}"
            } if self.ia else None
        }

    def update_confidence_threshold(self, threshold):
        """Update confidence threshold via WebSocket command"""
        if 0.1 <= threshold <= 0.9:
            self.current_confidence_threshold = threshold
            return {'success': True, 'threshold': threshold}
        return {'success': False, 'error': 'Threshold must be between 0.1 and 0.9'}

    def websocket_start_streaming(self, camera_index=0):
        """Start streaming specifically for WebSocket mode"""
        if self.is_streaming:
            return {'success': False, 'error': 'Already streaming'}
        
        try:
            # Connect to camera if not already connected
            if self.ia is None:
                if camera_index >= len(self.h.device_info_list):
                    return {'success': False, 'error': f'Camera index {camera_index} not available'}
                self.connect_camera(camera_index)
            
            self.start_streaming()
            return {'success': True, 'message': 'Streaming started'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def websocket_stop_streaming(self):
        """Stop streaming for WebSocket mode"""
        try:
            self.stop_streaming()
            return {'success': True, 'message': 'Streaming stopped'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ... [Keep all your existing methods: start_streaming, stop_streaming, etc.]

    def run_websocket_mode(self):
        """Run in WebSocket mode (no OpenCV display)"""
        print("Running in WebSocket mode - no GUI display")
        
        # Connect to first available camera by default
        if len(self.h.device_info_list) > 0:
            self.connect_camera(0)  # Use first camera
            print("Camera connected. Ready for WebSocket commands.")
        else:
            print("No cameras found. Waiting for commands...")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_streaming()
            print("WebSocket mode cleanup completed")

    def run(self):
        """Main run loop - chooses between GUI and WebSocket mode"""
        if self.websocket_mode:
            self.run_websocket_mode()
        else:
            self.run_gui_mode()

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
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    if self.is_streaming:
                        print("Stopping streaming...")
                        self.stop_streaming()
                    else:
                        print("Starting streaming...")
                        self.start_streaming()
                elif key == ord('c'):
                    self.current_confidence_threshold = max(0.1, self.current_confidence_threshold - 0.1)
                    print(f"Confidence threshold: {self.current_confidence_threshold:.1f}")
                elif key == ord('v'):
                    self.current_confidence_threshold = min(0.9, self.current_confidence_threshold + 0.1)
                    print(f"Confidence threshold: {self.current_confidence_threshold:.1f}")
                
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

    