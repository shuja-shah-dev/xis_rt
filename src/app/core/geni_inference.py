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
    def __init__(self, cti_file_path=None):
        if not HAS_TKINTER:
            raise RuntimeError("tkinter is not available. Please install tkinter.")
        
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
        
        # Initialize components
        # self.initialize_camera()
        # self.initialize_tensorrt()
        # self.setup_opencv_windows()

        self.initialize_camera(cti_file_path)
        self.initialize_tensorrt()
        if not self.websocket_mode:
            self.setup_opencv_windows()
            
    # def initialize_camera(self):
    #     """Initialize GenICam camera using Harvesters"""
    #     try:
    #         root = tk.Tk()
    #         root.withdraw()
            
    #         cti_path = filedialog.askopenfilename(
    #             title="Select GenTL Producer (.cti) File",
    #             filetypes=[("GenTL Producer", "*.cti"), ("All Files", "*.*")]
    #         )
            
    #         root.destroy()
            
    #         if not cti_path:
    #             raise RuntimeError("No CTI file selected")

    #         self.h = Harvester()
    #         self.h.add_file(cti_path)
    #         self.h.update()
            
    #         if len(self.h.device_info_list) == 0:
    #             raise RuntimeError("No cameras found")

    #         print(f"Found {len(self.h.device_info_list)} camera(s)")
    #         for i, device in enumerate(self.h.device_info_list):
    #             print(f"Camera {i}: {device}")
                
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to initialize camera: {e}")
    


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
    
    def list_available_cameras(self):
        """List all available cameras with details"""
        if len(self.h.device_info_list) == 0:
            print("No cameras found")
            return []
        
        print("Available cameras:")
        cameras = []
        for i, device in enumerate(self.h.device_info_list):
            vendor = getattr(device, 'vendor', 'Unknown')
            model = getattr(device, 'model', 'Unknown')
            serial = getattr(device, 'serial_number', 'Unknown')
            user_name = getattr(device, 'user_defined_name', 'Unknown')
            
            camera_info = {
                'index': i,
                'vendor': vendor,
                'model': model,
                'serial': serial,
                'user_name': user_name,
                'device': device
            }
            cameras.append(camera_info)
            
            print(f"  [{i}] {vendor} {model}")
            print(f"      Serial: {serial}")
            if user_name != 'Unknown':
                print(f"      User Name: {user_name}")
            print()
        
        return cameras

    def select_camera_interactive(self):
        """Interactive camera selection"""
        cameras = self.list_available_cameras()
        if not cameras:
            return None
        
        if len(cameras) == 1:
            print(f"Only one camera found. Using camera 0: {cameras[0]['vendor']} {cameras[0]['model']}")
            return 0
        
        while True:
            try:
                choice = input(f"Select camera (0-{len(cameras)-1}, or press Enter for camera 0): ").strip()
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

    def connect_camera(self, camera_index=None):
        """Connect to specific camera"""
        if camera_index is None:
            camera_index = self.select_camera_interactive()
            if camera_index is None:
                raise ValueError("No camera selected")
        
        if camera_index >= len(self.h.device_info_list):
            raise ValueError(f"Camera index {camera_index} not available. Found {len(self.h.device_info_list)} cameras.")
        
        self.ia = self.h.create(camera_index)
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
        format_priority = ["RGB8", "BGR8", "BayerRG8", "BayerGR8", "BayerBG8", "BayerGB8", "Mono8"]
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
        self.display_width, self.display_height = self._calculate_display_size(final_width, final_height)
        print(f"Display size: {self.display_width}x{self.display_height}")
        
    def _calculate_display_size(self, width, height, max_display_width=1920, max_display_height=1080):
        width_scale = max_display_width / width
        height_scale = max_display_height / height
        scale = min(width_scale, height_scale, 1.0)
        return int(width * scale), int(height * scale)

    def initialize_tensorrt(self):
        """Initialize TensorRT detector"""
        try:
            print("Initializing TensorRT...")
            t0 = time.perf_counter()
            engine_path = r"E:\Workspace\Eman\Vim X\models\largefp16.engine"
            if not os.path.exists(engine_path):
                print(f"Engine not found: {engine_path}")
                return

            self.ctx.push()
            self.detector = TensorRTDetector(engine_path, max_detections=1000)
            load_ms = (time.perf_counter() - t0) * 1e3
            print(f"TensorRT engine loaded in {load_ms:.1f} ms")
            self.ctx.pop()

            print("TensorRT engine loaded successfully")

        except Exception as e:
            print(f"TensorRT initialization error: {e}")
            
    def setup_opencv_windows(self):
        """Setup OpenCV window"""
        cv2.namedWindow(self.inference_window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.inference_window, 700, 50)
        cv2.resizeWindow(self.inference_window, self.display_width, self.display_height)
        print("OpenCV inference window created")

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

    def inference_worker(self):
        """TensorRT inference worker thread"""
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
                print(f"[POST] {post_ms:5.1f} ms")

                # Calculate display FPS
                now = time.perf_counter()
                done_timestamps.append(now)

                while done_timestamps and now - done_timestamps[0] > 1.0:
                    done_timestamps.popleft()

                disp_fps = len(done_timestamps)
                cv2.putText(vis, f"{disp_fps:5.0f} DISPLAY FPS", (10, 78),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                with self.detection_lock:
                    self.latest_detections = vis

                self.inference_queue.task_done()

        finally:
            self.ctx.pop()
            print("TensorRT inference worker stopped")

    def camera_capture_worker(self):
        """Camera capture worker thread"""
        try:
            self.ia.start()
            frame_count = 0
            
            while self.is_streaming:
                try:
                    with self.ia.fetch(timeout=2000) as buffer:
                        # Process frame to RGB format
                        frame = self.process_frame(buffer)
                        
                        # Convert RGB to BGR for OpenCV processing
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        current_time = time.perf_counter()
                        
                        # Update frame counting
                        frame_count += 1
                        if self.camera_start_time is None:
                            self.camera_start_time = current_time
                        
                        # Send to inference queue with letterboxing
                        if self.inference_running and not self.inference_queue.full():
                            # Letterbox to 640x640
                            h, w = frame_bgr.shape[:2]
                            scale = min(640 / h, 640 / w)
                            nw, nh = int(round(w * scale)), int(round(h * scale))
                            resized = cv2.resize(frame_bgr, (nw, nh))
                            canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
                            top, left = (640 - nh) // 2, (640 - nw) // 2
                            canvas[top:top + nh, left:left + nw] = resized
                            
                            try:
                                self.inference_queue.put_nowait((canvas, current_time))
                            except queue.Full:
                                pass
                        
                        # Send to display queue
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

    def start_streaming(self):
        """Start camera streaming with TensorRT inference"""
        if self.is_streaming:
            return
            
        try:
            if self.detector is None:
                print("TensorRT not initialized. Please restart the application.")
                return
            
            if self.ia is None:
                print("Camera not connected.")
                return
            
            # Reset counters
            self.camera_frame_count = 0
            self.camera_start_time = None
            
            # Start inference worker
            self.inference_running = True
            self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
            self.inference_thread.start()
            
            # Start camera capture worker
            self.is_streaming = True
            self.camera_thread = threading.Thread(target=self.camera_capture_worker, daemon=True)
            self.camera_thread.start()
            
            print("Streaming started with TensorRT inference...")
            
        except Exception as e:
            print(f"Failed to start streaming: {str(e)}")

    def stop_streaming(self):
        """Stop camera streaming and TensorRT inference"""
        if not self.is_streaming:
            return

        try:
            self.is_streaming = False
            self.inference_running = False

            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=2.0)

            if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2.0)

            if self.ia:
                try:
                    self.ia.stop()
                    self.ia.destroy()
                except:
                    pass
                self.ia = None

            if self.h:
                self.h.reset()

            # Clean up CUDA context
            if self.ctx is not None:
                try:
                    self.ctx.pop()
                except cuda.LogicError:
                    pass
                self.ctx.detach()
                self.ctx = None

            print("Camera streaming stopped")

        except Exception as e:
            print(f"Failed to stop streaming: {e}")

    def display_frames(self):
        """Display frames with inference results"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                with self.detection_lock:
                    frame_with_detections = self.latest_detections if self.latest_detections is not None else frame
                cv2.imshow(self.inference_window, frame_with_detections)
        except Exception as e:
            print(f"Display error: {e}")

    def run(self):
        """Main run loop"""
        print("Starting TensorRT GenICam Detection System")
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
            print("Cleanup completed")

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

    