import cv2
import base64
import threading
import time
from flask_socketio import emit

class GenICamService:
    def __init__(self):
        self.socketio = None
        self.streaming = False
        self.stream_thread = None
        self.conf_threshold = 0.5
        self.client_count = 0
        
    def set_socketio(self, socketio):
        self.socketio = socketio
        
    def client_joined(self):
        self.client_count += 1
        return self.client_count
        
    def client_left(self):
        self.client_count -= 1
        return self.client_count
        
    def set_confidence(self, value):
        self.conf_threshold = value
        return self.conf_threshold
    
    def run_normal(self):
        if not self.streaming:
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_normal_frames)
            self.stream_thread.daemon = True
            self.stream_thread.start()
    
    def run_with_inference(self, engine_path):
        if not self.streaming:
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_inference_frames, args=(engine_path,))
            self.stream_thread.daemon = True
            self.stream_thread.start()
    
    def stop(self):
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1)
    
    def _stream_normal_frames(self):
        cap = cv2.VideoCapture(0)  # Use your camera source here
        
        while self.streaming and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                if self.socketio:
                    self.socketio.emit('stream_frame', {
                        'frame': frame_b64,
                        'metrics': {'display_fps': 30}
                    }, room='stream')
                
                time.sleep(1/30)  # 30 FPS
        
        cap.release()
    
    def _stream_inference_frames(self, engine_path):
        cap = cv2.VideoCapture(0)  # Use your camera source here
        
        while self.streaming and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Add your inference logic here
                processed_frame = self._run_inference(frame, engine_path)
                
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                if self.socketio:
                    self.socketio.emit('inference_result', {
                        'frame': frame_b64,
                        'detections': []  # Add detection results
                    }, room='stream')
                
                time.sleep(1/30)  # 30 FPS
        
        cap.release()
    
    def _run_inference(self, frame, engine_path):
        # Add your actual inference logic here
        # For now, just return the original frame
        return frame