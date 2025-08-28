# app/core/camera_wrapper.py
"""
Camera wrapper that removes GUI dependencies for web server use
"""

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
import gc
import atexit
from typing import Dict, List, Optional
from collections import deque

# Import the detection classes from your original file
try:
    from app.core.geni_inference import (
        DetectionResult, Transform, LoadImageFromFile, Resize, Pad, 
        Normalize, DefaultFormatBundle, Collect, build_transform, 
        Pipeline, TensorRTDetector, visualize_detections_img
    )
except ImportError as e:
    print(f"Warning: Could not import from geni_inference: {e}")
    # You may need to define these locally if import fails

cuda.init()

class WebServerCameraDetector:
    """Camera detector optimized for web server use - no GUI dependencies"""
    
    def __init__(self, cti_file_path=None):
        # Camera variables
        self.h = None
        self.ia = None
        self.vendor = None
        self.pixel_format = None
        self.original_pixel_format = None
        
        # Resource management - MUST be initialized first
        self.shutdown_event = threading.Event()
        self.cleanup_callbacks = []
        
        # CUDA context management
        self.ctx = None
        self.cuda_device = None
        try:
            self.cuda_device = cuda.Device(0)
            self._init_cuda_context()
        except Exception as e:
            print(f"CUDA device initialization failed: {e}")
        
        # Streaming control
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.inference_queue = queue.Queue(maxsize=15)
        
        # TensorRT components
        self.detector = None
        self.inference_thread = None
        self.camera_thread = None
        self.inference_running = False
        self.latest_detections = None
        self.latest_detections_raw = []
        self.detection_lock = threading.RLock()
        
        # WebSocket integration
        self.websocket_mode = True  # Always true for web server
        self.websocket_server = None
        
        # Performance tracking
        self.camera_frame_count = 0
        self.camera_start_time = None
        self.current_confidence_threshold = 0.6
        
        # Display settings
        self.display_width = 1280
        self.display_height = 720
        
        # Initialize components
        if cti_file_path:
            self.initialize_camera_with_cti(cti_file_path)
        # Only initialize TensorRT if we have CUDA context
        if self.ctx is not None:
            self.initialize_tensorrt()
        
        # Register cleanup
        atexit.register(self._final_cleanup)
    
    def _init_cuda_context(self):
        """Initialize CUDA context with better error handling"""
        try:
            if self.cuda_device is not None:
                self.ctx = self.cuda_device.make_context()
                self.cleanup_callbacks.append(self._cleanup_cuda_context)
                print("CUDA context initialized successfully")
            else:
                print("No CUDA device available")
                self.ctx = None
        except Exception as e:
            print(f"CUDA context initialization failed: {e}")
            self.ctx = None
    
    def _cleanup_cuda_context(self):
        """Clean up CUDA context"""
        if self.ctx:
            try:
                self.ctx.pop()
                self.ctx.detach()
                self.ctx = None
                print("CUDA context cleaned up")
            except Exception as e:
                print(f"CUDA context cleanup error: {e}")
    
    def initialize_camera_with_cti(self, cti_file_path):
        """Initialize camera with specific CTI file - no GUI"""
        try:
            print(f"Initializing camera with CTI file: {cti_file_path}")
            
            # Verify CTI file exists
            if not os.path.exists(cti_file_path):
                raise RuntimeError(f"CTI file not found: {cti_file_path}")

            self.h = Harvester()
            self.h.add_file(cti_file_path)
            self.h.update()
            
            if len(self.h.device_info_list) == 0:
                raise RuntimeError("No cameras found with this CTI file")

            print(f"Found {len(self.h.device_info_list)} camera(s)")
            for i, device in enumerate(self.h.device_info_list):
                vendor = getattr(device, 'vendor', 'Unknown')
                model = getattr(device, 'model', 'Unknown')
                serial = getattr(device, 'serial_number', 'Unknown')
                print(f"Camera {i}: {vendor} {model} (S/N: {serial})")
                
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize camera: {e}")
    
    def connect_camera(self, camera_index=0):
        """Connect to specific camera"""
        try:
            if camera_index >= len(self.h.device_info_list):
                raise ValueError(f"Camera index {camera_index} not available. Found {len(self.h.device_info_list)} cameras.")
            
            print(f"Connecting to camera {camera_index}")
            self.ia = self.h.create(camera_index)
            node_map = self.ia.remote_device.node_map
            self.vendor = self.h.device_info_list[camera_index].vendor

            # Store original format
            self.original_pixel_format = node_map.PixelFormat.value
            print(f"Original pixel format: {self.original_pixel_format}")

            # Set maximum resolution
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
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            raise
    
    def _calculate_display_size(self, width, height, max_display_width=1920, max_display_height=1080):
        """Calculate appropriate display size"""
        width_scale = max_display_width / width
        height_scale = max_display_height / height
        scale = min(width_scale, height_scale, 1.0)
        return int(width * scale), int(height * scale)

    def initialize_tensorrt(self):
        """Initialize TensorRT detector - only if CUDA is available"""
        if self.ctx is None:
            print("Skipping TensorRT initialization - no CUDA context available")
            return

        try:
            print("Initializing TensorRT...")
            engine_path = r"E:\Workspace\Eman\Vim X\models\largefp16.engine"
            
            # Check if engine exists
            if not os.path.exists(engine_path):
                print(f"Engine not found: {engine_path}")
                print("TensorRT detector will be initialized later when inference is requested")
                return

            self.ctx.push()
            try:
                self.detector = TensorRTDetector(engine_path, max_detections=1000)
                print("TensorRT engine loaded successfully")
            finally:
                self.ctx.pop()

        except Exception as e:
            print(f"TensorRT initialization error: {e}")
            if self.ctx:
                try:
                    self.ctx.pop()
                except:
                    pass
    
    def load_tensorrt_engine(self, engine_path):
        """Load TensorRT engine for inference"""
        try:
            if not os.path.exists(engine_path):
                raise RuntimeError(f"Engine file not found: {engine_path}")
            
            print(f"Loading TensorRT engine: {engine_path}")
            
            if self.ctx:
                self.ctx.push()
                if self.detector:
                    # Clean up existing detector
                    del self.detector
                    gc.collect()
                
                self.detector = TensorRTDetector(engine_path, max_detections=1000)
                self.ctx.pop()
                print("TensorRT engine loaded successfully")
                return True
            else:
                print("CUDA context not available")
                return False
                
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            if self.ctx:
                try:
                    self.ctx.pop()
                except:
                    pass
            return False

    def process_frame(self, buffer):
        """Process frame from GenICam camera to RGB format"""
        component = buffer.payload.components[0]
        width, height, data = component.width, component.height, component.data

        # Handle different pixel formats
        if self.pixel_format == "RGB8":
            image = data.reshape((height, width, 3))
            if self.vendor and self.vendor.lower().startswith("allied vision"):
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
        if self.detector is None or self.ctx is None:
            print("TensorRT not initialized for inference.")
            return

        print("TensorRT inference worker running")
        
        try:
            self.ctx.push()
            done_timestamps = deque()
            last_cap_ts = None
            
            while self.inference_running and not self.shutdown_event.is_set():
                try:
                    try:
                        frame, cap_ts = self.inference_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if frame is None or self.shutdown_event.is_set():
                        continue

                    # Calculate FPS
                    if last_cap_ts is not None:
                        cam_fps = 1.0 / (cap_ts - last_cap_ts)
                    else:
                        cam_fps = 0
                    last_cap_ts = cap_ts

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
                    
                    # Add performance text
                    cv2.putText(vis, f"{disp_fps:5.0f} DISPLAY FPS", (10, 78),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(vis, f"GPU: {latency:.1f}ms", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Update results thread-safely
                    with self.detection_lock:
                        if self.latest_detections is not None:
                            del self.latest_detections
                        self.latest_detections = vis.copy()
                        self.latest_detections_raw = detections[:]

                    # Send to WebSocket if enabled
                    if self.websocket_mode and self.websocket_server:
                        metrics = {
                            'fps': disp_fps,
                            'gpu_latency_ms': latency,
                            'detection_count': len(detections),
                            'confidence_threshold': self.current_confidence_threshold
                        }
                        self.websocket_server.send_frame_to_websocket(vis, detections, metrics)

                    # Clean up
                    del frame, vis
                    self.inference_queue.task_done()

                except Exception as e:
                    print(f"Inference worker error: {e}")
                    time.sleep(0.01)

        except Exception as e:
            print(f"Inference worker fatal error: {e}")
        finally:
            self.ctx.pop()
            print("TensorRT inference worker stopped")

    def camera_capture_worker(self):
        """Camera capture worker thread"""
        if not self.ia:
            print("Camera not available")
            return
            
        try:
            self.ia.start()
            frame_count = 0
            
            while self.is_streaming and not self.shutdown_event.is_set():
                try:
                    with self.ia.fetch(timeout=1000) as buffer:
                        if self.shutdown_event.is_set():
                            break
                            
                        # Process frame
                        frame = self.process_frame(buffer)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        current_time = time.perf_counter()
                        
                        frame_count += 1
                        if self.camera_start_time is None:
                            self.camera_start_time = current_time
                        
                        # Send to inference queue if running
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
                            self.frame_queue.put(frame_bgr.copy())
                        
                        # Send to WebSocket for normal streaming
                        if (self.websocket_mode and self.websocket_server and 
                            not self.inference_running):
                            fps = (frame_count / (current_time - self.camera_start_time) 
                                  if self.camera_start_time else 0)
                            metrics = {
                                'fps': fps,
                                'frame_count': frame_count,
                                'inference_active': False
                            }
                            self.websocket_server.send_frame_to_websocket(
                                frame_bgr, None, metrics)
                        
                        # Clean up
                        del frame, frame_bgr
                        
                        if frame_count % 100 == 0:
                            print(f"Frame {frame_count} processed")
                            
                except Exception as e:
                    if self.is_streaming and not self.shutdown_event.is_set():
                        print(f"Frame capture error: {e}")
                        time.sleep(0.01)
                        
        except Exception as e:
            print(f"Camera worker error: {e}")
        finally:
            try:
                if self.ia:
                    self.ia.stop()
            except:
                pass
            print("Camera capture worker stopped")

    def start_streaming(self, with_inference=False, engine_path=None):
        """Start camera streaming"""
        if self.is_streaming:
            print("Already streaming")
            return True
            
        try:
            if with_inference:
                if engine_path and not self.load_tensorrt_engine(engine_path):
                    print("Failed to load TensorRT engine")
                    return False
                elif not self.detector:
                    print("No TensorRT detector available for inference")
                    return False
            
            if not self.ia:
                print("Camera not connected")
                return False
            
            # Reset shutdown event
            self.shutdown_event.clear()
            
            # Reset counters
            self.camera_frame_count = 0
            self.camera_start_time = None
            
            # Start inference worker if needed
            if with_inference:
                self.inference_running = True
                self.inference_thread = threading.Thread(
                    target=self.inference_worker,
                    daemon=False,
                    name="InferenceWorker"
                )
                self.inference_thread.start()
            
            # Start camera capture worker
            self.is_streaming = True
            self.camera_thread = threading.Thread(
                target=self.camera_capture_worker,
                daemon=False,
                name="CameraWorker"
            )
            self.camera_thread.start()
            
            mode = "with inference" if with_inference else "normal"
            print(f"Camera streaming started ({mode})")
            return True
            
        except Exception as e:
            print(f"Failed to start streaming: {e}")
            self.stop_streaming()
            return False

    def stop_streaming(self):
        """Stop camera streaming"""
        if not self.is_streaming:
            return True

        try:
            print("Stopping camera streaming...")
            
            # Signal shutdown
            self.is_streaming = False
            self.inference_running = False
            self.shutdown_event.set()

            # Wait for threads
            threads_to_join = []
            if hasattr(self, 'inference_thread') and self.inference_thread and self.inference_thread.is_alive():
                threads_to_join.append(("InferenceWorker", self.inference_thread))
            if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
                threads_to_join.append(("CameraWorker", self.camera_thread))

            for name, thread in threads_to_join:
                try:
                    thread.join(timeout=3.0)
                    if thread.is_alive():
                        print(f"Warning: {name} thread did not stop cleanly")
                except Exception as e:
                    print(f"Error joining {name} thread: {e}")

            # Stop camera
            if self.ia:
                try:
                    self.ia.stop()
                except Exception as e:
                    print(f"Camera stop error: {e}")

            # Clear queues
            for q in [self.frame_queue, self.inference_queue]:
                while not q.empty():
                    try:
                        item = q.get_nowait()
                        del item
                    except queue.Empty:
                        break

            # Clean up detection results
            with self.detection_lock:
                if self.latest_detections is not None:
                    del self.latest_detections
                    self.latest_detections = None
                self.latest_detections_raw.clear()

            # Force garbage collection
            gc.collect()
            print("Camera streaming stopped")
            return True

        except Exception as e:
            print(f"Failed to stop streaming: {e}")
            return False

    def _final_cleanup(self):
        """Final cleanup"""
        try:
            print("Performing camera system cleanup...")
            
            self.stop_streaming()
            
            if self.ia:
                try:
                    self.ia.destroy()
                except:
                    pass
                self.ia = None

            if self.h:
                try:
                    self.h.reset()
                except:
                    pass
                self.h = None
            
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Cleanup callback error: {e}")
            
            print("Camera system cleanup completed")
            
        except Exception as e:
            print(f"Camera cleanup error: {e}")

    def __del__(self):
        """Destructor"""
        try:
            self._final_cleanup()
        except:
            pass