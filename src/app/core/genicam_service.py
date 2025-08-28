# app/core/genicam_service.py
import threading
import time
import cv2
import base64
import queue
import numpy as np
import gc
import psutil
import os
import atexit
from collections import deque
from contextlib import contextmanager
import weakref

# Import your existing TensorRT detector classes
from app.core.geni_inference import TensorRTGenICamDetector, DetectionResult

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

class MemoryMonitor:
    """Monitor memory usage"""
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
    
    def check_memory(self):
        """Check if memory usage exceeds threshold"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            if memory_mb > self.threshold_mb:
                print(f"Warning: Memory usage {memory_mb:.1f}MB exceeds threshold {self.threshold_mb}MB")
                return False
            return True
        except Exception as e:
            print(f"Memory check error: {e}")
            return True  # Don't fail if we can't check memory
    
    def get_memory_info(self):
        """Get current memory usage"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            return {'memory_mb': memory_mb, 'threshold_mb': self.threshold_mb}
        except Exception as e:
            return {'memory_mb': 0, 'threshold_mb': self.threshold_mb, 'error': str(e)}

class GenICamService:
    def __init__(self):
        # Core camera system
        self.camera_system = None
        self.socketio = None
        self.app_config = None
        
        # Streaming state
        self.streaming_active = False
        self.inference_active = False
        self.current_confidence = 0.6
        
        # Client management
        self.connected_clients = 0
        self.client_lock = threading.Lock()
        
        # Resource management
        self.resource_manager = ResourceManager()
        self.memory_monitor = MemoryMonitor(threshold_mb=1500)
        self.shutdown_event = threading.Event()
        self.threads = []
        
        # Frame management
        self.frame_queue = queue.Queue(maxsize=5)
        self.websocket_thread = None
        self.frame_cache = deque(maxlen=10)
        
        # Performance tracking
        self.frame_count = 0
        self._memory_check_counter = 0
        
        # Register cleanup only (no signal handlers - they must be in main thread)
        atexit.register(self.cleanup_all)
        
        print("GenICamService initialized")
    
    def set_socketio(self, socketio_instance):
        """Set the SocketIO instance"""
        self.socketio = socketio_instance
        print("SocketIO instance set on GenICamService")
        
    def set_app_config(self, app_config):
        """Set the application configuration"""
        self.app_config = app_config
        print("App config set on GenICamService")
    
    # def _initialize_camera(self):
    #     """Initialize camera system with proper error handling"""
    #     try:
    #         if self.camera_system is None:
    #             print("Initializing TensorRT camera system...")
    #             self.camera_system = TensorRTGenICamDetector()
    #             self.resource_manager.register_resource(
    #                 self.camera_system, 
    #                 self._cleanup_camera_system
    #             )
                
    #             # Configure for WebSocket mode
    #             self.camera_system.websocket_mode = True
    #             self.camera_system.websocket_server = self
                
    #             # Connect to first available camera
    #             if len(self.camera_system.h.device_info_list) > 0:
    #                 self.camera_system.connect_camera(0)
    #                 print("Camera connected successfully")
    #             else:
    #                 print("No cameras found")
    #                 return False
    #             return True
    #         return True
    #     except Exception as e:
    #         print(f"Camera initialization failed: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         self._cleanup_camera_system()
    #         return False
    
        
    def _initialize_camera(self):
        """Initialize camera system - temporary fix using simple camera"""
        try:
            if self.camera_system is None:
                print("Initializing simple camera system (no CUDA)...")
                
                # Get CTI file path from app config
                cti_file_path = None
                if self.app_config:
                    config = self.app_config.get_config()
                    cti_file_path = config.get('cti_file_location')
                    
                    if cti_file_path:
                        if not os.path.isabs(cti_file_path):
                            cti_file_path = os.path.abspath(cti_file_path)
                        
                        print(f"Using CTI file from config: {cti_file_path}")
                        
                        if not os.path.exists(cti_file_path):
                            raise RuntimeError(f"CTI file not found: {cti_file_path}")
                    else:
                        raise RuntimeError("No CTI file specified in configuration")
                else:
                    raise RuntimeError("No app configuration available")
                
                # Use simple camera detector (no CUDA issues)
                try:
                    from app.core.simple_camera import SimpleCameraDetector
                    self.camera_system = SimpleCameraDetector(cti_file_path=cti_file_path)
                    print("Simple camera detector created successfully")
                except ImportError as e:
                    print(f"Could not import SimpleCameraDetector: {e}")
                    return False
                
                self.resource_manager.register_resource(
                    self.camera_system, 
                    self._cleanup_camera_system
                )
                
                # Configure for WebSocket mode
                self.camera_system.websocket_mode = True
                self.camera_system.websocket_server = self
                
                # Connect to camera
                if hasattr(self.camera_system, 'h') and len(self.camera_system.h.device_info_list) > 0:
                    self.camera_system.connect_camera(0)
                    print("Camera connected successfully")
                else:
                    print("No cameras found")
                    return False
                return True
            return True
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_camera_system()
            return False

    # Also update the run_normal method to use the simple streaming
    def run_normal(self):
        """Start normal camera stream - simple version"""
        try:
            print("Starting normal stream (simple camera)...")
            
            if not self._initialize_camera():
                print("Failed to initialize camera")
                return False
                
            self.streaming_active = True
            self.inference_active = False
            self.shutdown_event.clear()
            
            # Clear any existing frames
            self._clear_frame_queues()
            
            # Start simple camera streaming
            if hasattr(self.camera_system, 'start_streaming'):
                success = self.camera_system.start_streaming(with_inference=False)
            else:
                print("Camera system does not support start_streaming method")
                return False
            
            if success:
                # Start WebSocket frame sender
                self._start_websocket_thread()
                print("Normal stream started successfully (simple camera)")
                return True
            else:
                print("Failed to start camera streaming")
                return False
            
        except Exception as e:
            print(f"Failed to start normal stream: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
            return False
    def _cleanup_camera_system(self):
        """Clean up camera system resources"""
        if self.camera_system:
            try:
                print("Cleaning up camera system...")
                self.camera_system.stop_streaming()
                if hasattr(self.camera_system, 'ctx') and self.camera_system.ctx:
                    try:
                        self.camera_system.ctx.pop()
                        self.camera_system.ctx.detach()
                    except:
                        pass
                self.camera_system = None
                print("Camera system cleaned up")
            except Exception as e:
                print(f"Camera cleanup error: {e}")
        
        
    # def run_normal(self):
    #     """Start normal camera stream"""
    #     try:
    #         print("Starting normal stream...")
            
    #         if not self._initialize_camera():
    #             print("Failed to initialize camera")
    #             return False
                
    #         self.streaming_active = True
    #         self.inference_active = False
    #         self.shutdown_event.clear()
            
    #         # Clear any existing frames
    #         self._clear_frame_queues()
            
    #         # Start camera system - use new method if available
    #         if hasattr(self.camera_system, 'start_streaming'):
    #             success = self.camera_system.start_streaming(with_inference=False)
    #         else:
    #             # Fallback to old method
    #             self.camera_system.start_streaming()
    #             success = True
            
    #         if success:
    #             # Start WebSocket frame sender
    #             self._start_websocket_thread()
    #             print("Normal stream started successfully")
    #             return True
    #         else:
    #             print("Failed to start camera streaming")
    #             return False
            
    #     except Exception as e:
    #         print(f"Failed to start normal stream: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         self.stop()
    #         return False

    def run_with_inference(self, engine_path):
        """Start camera stream with inference"""
        try:
            print(f"Starting inference stream with engine: {engine_path}")
            
            if not self._initialize_camera():
                print("Failed to initialize camera")
                return False
            
            # Verify engine path exists
            if not os.path.exists(engine_path):
                print(f"Engine file not found: {engine_path}")
                return False
                
            self.streaming_active = True
            self.inference_active = True
            self.shutdown_event.clear()
            
            # Clear any existing frames
            self._clear_frame_queues()
            
            # Update confidence threshold
            self.camera_system.current_confidence_threshold = self.current_confidence
            
            # Start camera system with inference - use new method if available
            if hasattr(self.camera_system, 'start_streaming'):
                success = self.camera_system.start_streaming(with_inference=True, engine_path=engine_path)
            else:
                # Fallback to old method
                if hasattr(self.camera_system, 'load_tensorrt_engine'):
                    self.camera_system.load_tensorrt_engine(engine_path)
                self.camera_system.start_streaming()
                success = True
            
            if success:
                # Start WebSocket frame sender
                self._start_websocket_thread()
                print("Inference stream started successfully")
                return True
            else:
                print("Failed to start camera streaming with inference")
                return False
            
        except Exception as e:
            print(f"Failed to start inference stream: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
            return False

    def _start_websocket_thread(self):
        """Start WebSocket thread with proper management"""
        if not self.websocket_thread or not self.websocket_thread.is_alive():
            self.websocket_thread = threading.Thread(
                target=self._websocket_frame_sender,
                daemon=False,
                name="GenICamWebSocketSender"
            )
            self.websocket_thread.start()
            self.threads.append(self.websocket_thread)
            print("WebSocket sender thread started")
    
        
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
                if hasattr(self.camera_system, 'stop_streaming'):
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
                self.socketio.emit('status', {
                    'message': 'Stream stopped',
                    'streaming_active': False
                }, namespace='/ws', to='stream')
            
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
            if hasattr(self.camera_system, 'frame_queue'):
                queues_to_clear.append(self.camera_system.frame_queue)
            if hasattr(self.camera_system, 'inference_queue'):
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
    
    def send_frame_to_websocket(self, frame, detections=None, metrics=None):
        """Send frame to WebSocket clients with debug logging"""
        if not self.socketio or not self.streaming_active:
            print("[DEBUG] Cannot send frame - no socketio or not streaming")
            return
        
        try:
            # Check memory usage
            if not self.memory_monitor.check_memory():
                print("[DEBUG] Skipping frame due to memory usage")
                return  # Skip frame to prevent memory issues
            
            # Encode frame to base64
            frame_copy = frame.copy() if not frame.flags.writeable else frame
            _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clean up intermediate variables
            del buffer
            if frame_copy is not frame:
                del frame_copy
            
            # Prepare data
            data = {
                'frame': frame_base64,
                'timestamp': time.time(),
                'metrics': metrics or {},
                'memory_info': self.memory_monitor.get_memory_info()
            }
            
            if detections is not None:
                # Convert detections to serializable format
                detection_data = []
                for det in detections:
                    detection_data.append({
                        'bbox': det.bbox,
                        'label_id': det.label_id,
                        'score': det.score
                    })
                data['detections'] = detection_data
                
                # Send as inference result
                self.socketio.emit('inference_result', data, namespace='/ws', to='stream')
                print(f"[DEBUG] Sent inference_result with {len(detection_data)} detections")
            else:
                # Send as normal stream frame
                self.socketio.emit('stream_frame', data, namespace='/ws', to='stream')
                frame_count = metrics.get('frame_send_count', 0) if metrics else 0
                if frame_count % 30 == 0:  # Debug every 30th frame
                    print(f"[DEBUG] Sent stream_frame #{frame_count}, frame size: {len(frame_base64)} chars")
            
            # Clean up
            del frame_base64
            del data
            
        except Exception as e:
            print(f"[ERROR] Error sending frame to WebSocket: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Periodic cleanup
            self.frame_count += 1
            if self.frame_count % 50 == 0:
                gc.collect()    
            
    def _websocket_frame_sender(self):
        """WebSocket frame sender thread with debug logging"""
        print("GenICam WebSocket frame sender started")
        frame_send_count = 0
        
        try:
            while self.streaming_active and not self.shutdown_event.is_set():
                try:
                    if self.camera_system and hasattr(self.camera_system, 'frame_queue'):
                        try:
                            frame = self.camera_system.frame_queue.get(timeout=0.1)
                            
                            if frame_send_count % 30 == 0:  # Debug every 30th frame
                                print(f"[DEBUG] WebSocket sender - Frame {frame_send_count}")
                                print(f"[DEBUG] Frame shape: {frame.shape if hasattr(frame, 'shape') else 'No shape'}")
                                print(f"[DEBUG] SocketIO available: {self.socketio is not None}")
                                print(f"[DEBUG] Streaming active: {self.streaming_active}")
                            
                        except queue.Empty:
                            continue
                        
                        if frame is None:
                            continue
                        
                        # Try to send frame to WebSocket
                        try:
                            if self.inference_active:
                                # Get latest detections (not applicable for simple camera)
                                detections = None
                                self.send_frame_to_websocket(frame, detections, {
                                    'inference_active': True,
                                    'confidence_threshold': self.current_confidence,
                                    'connected_clients': self.connected_clients
                                })
                            else:
                                # Send normal frame
                                self.send_frame_to_websocket(frame, None, {
                                    'inference_active': False,
                                    'connected_clients': self.connected_clients,
                                    'frame_send_count': frame_send_count
                                })
                            
                            frame_send_count += 1
                            
                            if frame_send_count % 30 == 0:  # Debug every 30th frame  
                                print(f"[DEBUG] Sent frame {frame_send_count} to WebSocket")
                                
                        except Exception as e:
                            print(f"[ERROR] Failed to send frame to WebSocket: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # Clean up frame reference
                        del frame
                    else:
                        if frame_send_count == 0:  # Only print once
                            print("[DEBUG] Camera system or frame_queue not available")
                            frame_send_count = 1  # Prevent spam
                    
                    time.sleep(1/30)  # ~30 FPS
                    
                except Exception as e:
                    print(f"WebSocket sender error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"WebSocket sender thread error: {e}")
        finally:
            print("GenICam WebSocket frame sender stopped")
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
    
    def _auto_stop_if_no_clients(self):
        """Auto-stop stream if no clients are connected"""
        with self.client_lock:
            if self.connected_clients == 0 and self.streaming_active:
                print("Auto-stopping stream due to no connected clients")
                self.stop()
    
    def set_confidence(self, value):
        """Set confidence threshold"""
        try:
            self.current_confidence = max(0.1, min(0.9, float(value)))
            if self.camera_system:
                self.camera_system.current_confidence_threshold = self.current_confidence
            print(f"Confidence threshold set to: {self.current_confidence}")
            return self.current_confidence
        except Exception as e:
            print(f"Error setting confidence: {e}")
            return self.current_confidence
    
    @property
    def conf_threshold(self):
        """Get current confidence threshold"""
        return self.current_confidence
    
    def get_status(self):
        """Get current service status"""
        memory_info = self.memory_monitor.get_memory_info()
        return {
            'streaming_active': self.streaming_active,
            'inference_active': self.inference_active,
            'connected_clients': self.connected_clients,
            'confidence_threshold': self.current_confidence,
            'camera_connected': self.camera_system is not None,
            'memory_usage_mb': memory_info.get('memory_mb', 0),
            'memory_threshold_mb': memory_info.get('threshold_mb', 1500)
        }
    
    def cleanup_all(self):
        """Comprehensive cleanup of all resources"""
        print("Starting GenICam service cleanup...")
        
        try:
            # Stop streaming
            self.stop()
            
            # Clean up resource manager
            self.resource_manager.cleanup_all()
            
            # Reset client count
            with self.client_lock:
                self.connected_clients = 0
            
            # Final garbage collection
            gc.collect()
            
            print("GenICam service cleanup completed")
            
        except Exception as e:
            print(f"Error during GenICam cleanup: {e}")
    
    def __del__(self):
        """Destructor with cleanup"""
        try:
            self.cleanup_all()
        except:
            pass