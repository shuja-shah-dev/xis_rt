# app/core/simple_camera.py
"""
Simple camera wrapper - no CUDA/TensorRT to avoid crashes
"""

import threading
import time
import cv2
import queue
import numpy as np
import os
import gc
from collections import deque
from harvesters.core import Harvester

class SimpleCameraDetector:
    """Simple camera detector without CUDA/TensorRT - for WebSocket testing"""
    
    def __init__(self, cti_file_path=None):
        # Camera variables
        self.h = None
        self.ia = None
        self.vendor = None
        self.pixel_format = None
        
        # Streaming control
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Threads
        self.camera_thread = None
        
        # WebSocket integration
        self.websocket_mode = True
        self.websocket_server = None
        
        # Performance tracking
        self.camera_frame_count = 0
        self.camera_start_time = None
        self.current_confidence_threshold = 0.6
        
        # Resource management
        self.shutdown_event = threading.Event()
        
        # Initialize camera
        if cti_file_path:
            self.initialize_camera_with_cti(cti_file_path)
    
    def initialize_camera_with_cti(self, cti_file_path):
        """Initialize camera with CTI file"""
        try:
            print(f"Initializing camera with CTI file: {cti_file_path}")
            
            if not os.path.exists(cti_file_path):
                raise RuntimeError(f"CTI file not found: {cti_file_path}")

            self.h = Harvester()
            self.h.add_file(cti_file_path)
            self.h.update()
            
            if len(self.h.device_info_list) == 0:
                raise RuntimeError("No cameras found")

            print(f"Found {len(self.h.device_info_list)} camera(s)")
            for i, device in enumerate(self.h.device_info_list):
                vendor = getattr(device, 'vendor', 'Unknown')
                model = getattr(device, 'model', 'Unknown')
                print(f"Camera {i}: {vendor} {model}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")
    
    def connect_camera(self, camera_index=0):
        """Connect to camera"""
        try:
            self.ia = self.h.create(camera_index)
            node_map = self.ia.remote_device.node_map
            self.vendor = self.h.device_info_list[camera_index].vendor
            
            # Set pixel format
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
                        
            self.pixel_format = selected_format or node_map.PixelFormat.value
            
            # Set resolution
            max_width = node_map.Width.max
            max_height = node_map.Height.max
            node_map.Width.value = max_width
            node_map.Height.value = max_height
            
            print(f"Connected to camera: {max_width}x{max_height}, Format: {self.pixel_format}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to connect to camera: {e}")

    def process_frame(self, buffer):
        """Process frame from camera"""
        component = buffer.payload.components[0]
        width, height, data = component.width, component.height, component.data

        # Handle different pixel formats
        if self.pixel_format == "RGB8":
            image = data.reshape((height, width, 3))
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
            # Fallback
            gray = data.reshape((height, width))
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image.copy()

    def camera_capture_worker(self):
        """Camera capture worker"""
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
                            
                        frame = self.process_frame(buffer)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        current_time = time.perf_counter()
                        
                        frame_count += 1
                        if self.camera_start_time is None:
                            self.camera_start_time = current_time
                        
                        # Send to display queue
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame_bgr.copy())
                        
                        # Don't send to WebSocket here - let GenICamService handle it
                        # The GenICamService WebSocket sender thread will read from frame_queue
                        
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
                self.ia.stop()
            except:
                pass
            print("Camera capture stopped")

    def start_streaming(self, with_inference=False, engine_path=None):
        """Start streaming"""
        if self.is_streaming:
            return True
            
        if not self.ia:
            print("Camera not connected")
            return False
        
        try:
            self.shutdown_event.clear()
            self.camera_frame_count = 0
            self.camera_start_time = None
            
            self.is_streaming = True
            self.camera_thread = threading.Thread(
                target=self.camera_capture_worker,
                daemon=False,
                name="SimpleCameraWorker"
            )
            self.camera_thread.start()
            
            print("Simple camera streaming started")
            return True
            
        except Exception as e:
            print(f"Failed to start streaming: {e}")
            return False

    def stop_streaming(self):
        """Stop streaming"""
        if not self.is_streaming:
            return True

        try:
            print("Stopping simple camera streaming...")
            
            self.is_streaming = False
            self.shutdown_event.set()

            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=3.0)

            # Clear queue
            while not self.frame_queue.empty():
                try:
                    item = self.frame_queue.get_nowait()
                    del item
                except queue.Empty:
                    break

            print("Simple camera streaming stopped")
            return True

        except Exception as e:
            print(f"Failed to stop streaming: {e}")
            return False

    def cleanup_all(self):
        """Cleanup"""
        try:
            self.stop_streaming()
            
            if self.ia:
                self.ia.destroy()
                self.ia = None

            if self.h:
                self.h.reset()
                self.h = None
                
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        try:
            self.cleanup_all()
        except:
            pass