import argparse
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque, defaultdict
import json
import os
import threading
import queue
import base64
import csv

PINNED_THRESHOLD_BYTES = 16 * 1024 * 1024
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640


def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def unletterbox_boxes(boxes_xyxy, scale, left, top, out_w, out_h):
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / scale, 0, out_w - 1)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top) / scale, 0, out_h - 1)
    return out


def unletterbox_masks_with_timing(masks, scale, left, top, out_w, out_h):
    if masks.size == 0:
        return masks, 0.0, 0.0
    t0 = time.perf_counter()
    new_w, new_h = int(round(out_w * scale)), int(round(out_h * scale))
    cropped = masks[:, top:top + new_h, left:left + new_w]
    t1 = time.perf_counter()
    out_masks = np.zeros((cropped.shape[0], out_h, out_w), dtype=masks.dtype)
    t_r0 = time.perf_counter()
    for i in range(cropped.shape[0]):
        out_masks[i] = cv2.resize(cropped[i], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    t_r1 = time.perf_counter()
    return out_masks, (t1 - t0), (t_r1 - t_r0)


def draw_inference_region(frame, region):
    """Draw the inference region rectangle"""
    x, y, w, h = region
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)


def filter_by_class_thresholds(boxes, labels, scores, masks, class_thresholds):
    """Filter detections using per-class confidence thresholds"""
    if len(boxes) == 0:
        return boxes, labels, scores, masks, np.array([])
    
    keep_mask = np.zeros(len(scores), dtype=bool)
    
    for class_id, threshold in class_thresholds.items():
        class_mask = (labels == class_id) & (scores >= threshold)
        keep_mask |= class_mask
    
    if not np.any(keep_mask):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    keep_indices = np.flatnonzero(keep_mask)
    return (boxes[keep_mask], labels[keep_mask], scores[keep_mask], 
            masks[keep_mask] if masks.size > 0 else masks, keep_indices)


def is_point_in_region(point, region):
    """Check if a point is inside the inference region"""
    px, py = point
    rx, ry, rw, rh = region
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def filter_detections_in_region(boxes, labels, scores, masks, region):
    """Filter detections to only those in the inference region"""
    if len(boxes) == 0:
        return boxes, labels, scores, masks
    
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    in_region = []
    
    for i, center in enumerate(centers):
        if is_point_in_region(center, region):
            in_region.append(i)
    
    if not in_region:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    in_region = np.array(in_region)
    return boxes[in_region], labels[in_region], scores[in_region], masks[in_region]


def match_holes_to_doughnuts(doughnut_boxes, doughnut_labels, hole_boxes):
    """Match each doughnut (class 0,1) to all holes (class 2) inside it"""
    matches = []
    
    for i, (d_box, d_label) in enumerate(zip(doughnut_boxes, doughnut_labels)):
        if d_label not in [0, 1]:
            continue
        
        for j, h_box in enumerate(hole_boxes):
            h_center = (h_box[:2] + h_box[2:4]) / 2
            
            if (d_box[0] <= h_center[0] <= d_box[2] and 
                d_box[1] <= h_center[1] <= d_box[3]):
                matches.append((i, j))
    
    return matches


def calculate_measurements(doughnut_box, hole_box):
    """Calculate distances from hole center to doughnut top and bottom"""
    hole_center_y = (hole_box[1] + hole_box[3]) / 2
    top_distance = hole_center_y - doughnut_box[1]
    bottom_distance = doughnut_box[3] - hole_center_y
    
    return top_distance, bottom_distance


def draw_measurement_lines(frame, doughnut_box, hole_box, color=(255, 255, 255)):
    """Draw white lines from doughnut edges to hole edges"""
    d_x1, d_y1, d_x2, d_y2 = [int(x) for x in doughnut_box]
    h_x1, h_y1, h_x2, h_y2 = [int(x) for x in hole_box]
    
    hole_center_x = int((h_x1 + h_x2) / 2)
    
    cv2.line(frame, (hole_center_x, d_y1), (hole_center_x, h_y1), color, 2)
    cv2.line(frame, (hole_center_x, h_y2), (hole_center_x, d_y2), color, 2)


def draw_doughnut_numbers(frame, boxes, labels, region):
    """Number doughnuts from left to right in the region"""
    if len(boxes) == 0:
        return []
    
    doughnut_indices = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if label in [0, 1]:
            center = (box[:2] + box[2:4]) / 2
            if is_point_in_region(center, region):
                doughnut_indices.append((i, box[0]))
    
    doughnut_indices.sort(key=lambda x: x[1])
    
    numbered_doughnuts = []
    for num, (idx, _) in enumerate(doughnut_indices, 1):
        box = boxes[idx]
        label = labels[idx]
        
        x1, y1, x2, y2 = [int(v) for v in box]
        
        text_x = max(0, x1 - 5)
        text_y = max(20, y1 - 5)
        
        text_size = cv2.getTextSize(str(num), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, 
                     (text_x - 2, text_y - text_size[1] - 4),
                     (text_x + text_size[0] + 2, text_y + 4),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, str(num), (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        numbered_doughnuts.append((num, idx, label))
    
    return numbered_doughnuts


def colorize_doughnut_masks(frame, masks, boxes, labels, alpha=0.7):
    """Draw only red (bad) and green (good) masks for doughnuts"""
    if masks.size == 0:
        return frame
    
    overlay = frame.copy()
    
    for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        if label == 0:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        else:
            continue
        
        mask_bool = mask > 127
        overlay[mask_bool] = color
    
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


class TimeBasedRowTracker:
    def __init__(self, start_time=4.5, interval=1.0):
        """Track rows based on video timestamps"""
        self.start_time = start_time
        self.interval = interval
        self.processed_rows = set()
        self.current_row_number = 0
        self.last_processed_time = 0
    
    def should_process_row(self, current_time_seconds):
        """Check if current timestamp should trigger a new row processing"""
        if current_time_seconds < self.start_time:
            return False, None
        
        time_since_start = current_time_seconds - self.start_time
        current_row_interval = int(time_since_start / self.interval)
        expected_trigger_time = self.start_time + (current_row_interval * self.interval)
        
        if (current_time_seconds >= expected_trigger_time and 
            current_row_interval not in self.processed_rows and
            current_time_seconds > self.last_processed_time + 0.5):
            
            self.processed_rows.add(current_row_interval)
            self.current_row_number += 1
            self.last_processed_time = current_time_seconds
            
            print(f"Triggering row detection at {current_time_seconds:.2f}s (expected: {expected_trigger_time:.1f}s)")
            return True, self.current_row_number
        
        return False, None


class VideoInferenceService_dont:
    def __init__(self, websocket_mode=True):
        self.websocket_mode = websocket_mode
        self.socketio = None
        self.mqtt_client = None
        self.is_running = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.config = {}
        self.current_fps = 0
        self.last_websocket_frame_time = 0
        self.websocket_frame_interval = 1.0 / 30
        self.shutdown_event = threading.Event()
        self.row_counter = 0
        self.inference_active = False
        self.seg = None
        self.cap = None
        self.initialization_error = None
        self.force_stop_event = threading.Event()
        self.last_activity_time = time.time()
        self.processing_timeout = 60.0
        self.watchdog_thread = None
        self.emergency_stop_event = threading.Event()
        self.csv_data = []
        self.row_tracker = None

    def set_socketio(self, socketio):
        self.socketio = socketio

    def set_mqtt(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def initialize_from_config(self, config):
        self.config = {
            "engine_path": config.get("engine_path", ""),
            "video_path": config.get("video_path", ""),
            "score_threshold": config.get("score_threshold", 0.4),
            "class0_threshold": config.get("class0_threshold", None),
            "class1_threshold": config.get("class1_threshold", None),
            "class2_threshold": config.get("class2_threshold", None),
            "mask_threshold": config.get("mask_threshold", 0.4),
            "canvas_size": config.get("canvas_size", 640),
            "alpha": config.get("alpha", 0.7),
            "target_fps": config.get("target_fps", 30.0),
            "mqtt_topic": config.get("mqtt_topic", "detection/results"),
            "inference_height": config.get("inference_height", 120),
            "row_start_time": config.get("row_start_time", 4.5),
            "row_interval": config.get("row_interval", 1.0),
            "csv_output_dir": config.get("csv_output_dir", None),
        }

        if not os.path.exists(self.config["engine_path"]):
            print(f"Engine path does not exist: {self.config['engine_path']}")
            return False

        return True

    def _initialize_engine_in_thread(self):
        """Initialize the TensorRT engine in the processing thread"""
        try:
            print("Initializing TensorRT engine in processing thread...")
            self.seg = TRTSegmentor(self.config["engine_path"])
            print("TensorRT engine initialized successfully")
            return True
        except Exception as e:
            print(f"Engine initialization error in processing thread: {e}")
            self.initialization_error = str(e)
            return False

    def start_processing(self):
        if self.is_running:
            return False

        if not self.config.get("video_path") or not self.config.get("engine_path"):
            return False

        test_cap = cv2.VideoCapture(self.config["video_path"])
        if not test_cap.isOpened():
            print(f"Could not open video file: {self.config['video_path']}")
            return False
        test_cap.release()

        self.is_running = True
        self.shutdown_event.clear()
        self.force_stop_event.clear()
        self.emergency_stop_event.clear()
        self.initialization_error = None

        self.processing_thread = threading.Thread(
            target=self._process_video, daemon=True
        )
        self.processing_thread.start()

        self.watchdog_thread = threading.Thread(
            target=self._watchdog_monitor, daemon=True
        )
        self.watchdog_thread.start()

        time.sleep(0.1)
        if self.initialization_error:
            print(f"Initialization failed: {self.initialization_error}")
            self.stop_processing()
            return False

        return True

    def stop_processing(self):
        print("Stopping video processing...")
        self.is_running = False
        self.shutdown_event.set()
        self.force_stop_event.set()
        self.emergency_stop_event.set()

        if self.watchdog_thread and self.watchdog_thread.is_alive():
            print("Stopping watchdog thread...")
            self.watchdog_thread.join(timeout=1.0)

        if self.processing_thread and self.processing_thread.is_alive():
            print("Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=3.0)

            if self.processing_thread.is_alive():
                print("WARNING: Processing thread did not terminate gracefully")

        self.reset_to_initial_state()
        print("Video processing stopped and service reset")
        return True

    def _add_csv_data(self, row_number, doughnut_number, status, measurement_difference):
        """Add donut data to CSV collection"""
        csv_row = {
            'row_number': row_number,
            'doughnut_number': doughnut_number,
            'status': status,
            'measurement_difference': measurement_difference
        }
        self.csv_data.append(csv_row)
        print(f"Added to CSV: {csv_row}")

    def _save_csv_data(self):
        """Save CSV data to file"""
        if not self.csv_data:
            print("No CSV data to save")
            return False

        try:
            output_dir = self.config.get("csv_output_dir", "output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            csv_filename = f"doughnut_measurements_{timestamp_str}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)

            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['row_number', 'doughnut_number', 'status', 'measurement_difference']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data)

            print(f"CSV saved successfully: {csv_filepath}")
            print(f"Total rows: {len(self.csv_data)}")

            if self.socketio:
                self.socketio.emit(
                    "csv_saved",
                    {
                        "filepath": csv_filepath,
                        "filename": csv_filename,
                        "row_count": len(self.csv_data),
                    },
                    namespace="/ws",
                )

            return True

        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False

    def get_csv_data(self):
        """Return current CSV data"""
        return {"data": self.csv_data, "row_count": len(self.csv_data)}

    def emergency_shutdown(self):
        """Emergency stop with CSV saving"""
        print("Emergency shutdown initiated")

        self.is_running = False
        self.emergency_stop_event.set()
        self.force_stop_event.set()
        self.shutdown_event.set()

        try:
            self._save_csv_data()
        except:
            pass

        self.reset_to_initial_state()

        if self.socketio:
            self.socketio.emit(
                "video_status",
                {"status": "emergency_stopped", "reason": "system_protection"},
                namespace="/ws",
            )

        print("Emergency shutdown completed")
        return True

    def _watchdog_monitor(self):
        """Watchdog monitor for detecting system freezes"""
        print("Watchdog monitor started")

        while self.is_running and not self.emergency_stop_event.is_set():
            try:
                current_time = time.time()
                time_since_activity = current_time - self.last_activity_time

                if time_since_activity > self.processing_timeout:
                    print(f"MAJOR FREEZE DETECTED - EMERGENCY SHUTDOWN after {time_since_activity:.1f}s!")
                    self.emergency_shutdown()
                    break

                time.sleep(10.0)

            except Exception as e:
                print(f"Watchdog error: {e}")
                continue

        print("Watchdog monitor stopped")

    def reset_to_initial_state(self):
        """Reset the service back to initial state"""
        print("Resetting VideoInferenceService to initial state...")

        try:
            if self.cap:
                self.cap.release()
                self.cap = None
                print("Video capture released")
        except Exception as e:
            print(f"Warning: Error releasing video capture: {e}")

        self.is_running = False
        self.processing_thread = None
        self.current_fps = 0
        self.last_websocket_frame_time = 0
        self.row_counter = 0
        self.inference_active = False
        self.initialization_error = None
        self.last_activity_time = time.time()
        self.row_tracker = None

        self.shutdown_event.clear()
        self.force_stop_event.clear()

        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass

        self.seg = None
        print("Service reset to initial state completed")

    def force_stop(self):
        """Emergency stop function"""
        print("EMERGENCY STOP: Force stopping video processing...")
        self.is_running = False
        self.shutdown_event.set()
        self.force_stop_event.set()

        self.reset_to_initial_state()

        if self.socketio:
            self.socketio.emit(
                "video_status",
                {"status": "force_stopped", "reason": "emergency_stop"},
                namespace="/ws",
            )

        return True

    def force_cleanup(self):
        """Force cleanup of resources"""
        self.reset_to_initial_state()

    def is_stuck(self):
        """Check if processing appears to be stuck"""
        return (time.time() - self.last_activity_time) > self.processing_timeout

    def _process_video(self):
        """Main video processing loop - runs in separate thread"""
        print("Starting video processing thread...")

        if not self._initialize_engine_in_thread():
            print("Failed to initialize engine in processing thread")
            self.is_running = False
            return

        self.cap = cv2.VideoCapture(self.config["video_path"])
        if not self.cap.isOpened():
            print("Failed to open video capture in processing thread")
            self.is_running = False
            return

        try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            region_height = self.config["inference_height"]
            region_width = width
            region_x = 0
            region_y = (height - region_height) // 2
            inference_region = (region_x, region_y, region_width, region_height)

            self.row_tracker = TimeBasedRowTracker(
                start_time=self.config["row_start_time"],
                interval=self.config["row_interval"]
            )

            fps_counter = FPSCounter()
            frame_idx = 0

            class_thresholds = {
                0: self.config.get("class0_threshold") or self.config["score_threshold"],
                1: self.config.get("class1_threshold") or self.config["score_threshold"],
                2: self.config.get("class2_threshold") or self.config["score_threshold"],
            }

            print("Video processing loop started")
            print(f"Row detection starts at {self.config['row_start_time']}s, every {self.config['row_interval']}s")

            while self.is_running and not self.force_stop_event.is_set():
                try:
                    self.last_activity_time = time.time()

                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video reached")
                        self._save_csv_data()
                        if self.socketio:
                            self.socketio.emit(
                                "video_ended",
                                {"reason": "end_of_video"},
                                namespace="/ws",
                            )
                        break

                    current_video_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    current_video_time_seconds = current_video_time_ms / 1000.0

                    vis = self._process_frame(
                        frame, current_video_time_seconds, inference_region, 
                        class_thresholds, frame_idx, frame_count
                    )

                    fps_counter.update()
                    self.current_fps = fps_counter.get_fps()

                    self._add_fps_overlay(vis, width)
                    self._add_info_overlay(vis, current_video_time_seconds, frame_idx, frame_count, width)

                    self._emit_frame(
                        vis,
                        {
                            "timestamp": current_video_time_seconds,
                            "fps": self.current_fps,
                            "frame_idx": frame_idx,
                            "row_number": self.row_tracker.current_row_number if self.row_tracker else 0,
                        },
                    )

                    frame_idx += 1

                    if self.shutdown_event.is_set() or self.force_stop_event.is_set():
                        print("Shutdown event received")
                        break

                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    frame_idx += 1
                    continue

        except Exception as e:
            print(f"Critical error in video processing loop: {e}")
            import traceback
            traceback.print_exc()

            if self.socketio:
                self.socketio.emit(
                    "video_ended",
                    {"reason": "critical_error", "error": str(e)},
                    namespace="/ws",
                )

        finally:
            print("Cleaning up video processing thread...")
            if self.cap:
                self.cap.release()
                self.cap = None

            if self.seg:
                try:
                    print("Cleaning up TensorRT engine and CUDA context...")

                    if hasattr(self.seg, "context") and self.seg.context:
                        print("Destroying TensorRT execution context...")
                        del self.seg.context
                        self.seg.context = None

                    if hasattr(self.seg, "engine") and self.seg.engine:
                        print("Destroying TensorRT engine...")
                        del self.seg.engine
                        self.seg.engine = None

                    if hasattr(self.seg, "ctx") and self.seg.ctx:
                        print("Cleaning up CUDA context stack...")
                        context_count = 0
                        while True:
                            try:
                                self.seg.ctx.pop()
                                context_count += 1
                                print(f"Popped context #{context_count}")
                            except Exception:
                                print(f"No more contexts to pop (popped {context_count} total)")
                                break

                        try:
                            self.seg.ctx.detach()
                            print("CUDA context detached")
                        except Exception as detach_error:
                            print(f"Context detach not needed or failed: {detach_error}")

                        print("CUDA context stack cleaned successfully")

                except Exception as e:
                    print(f"Warning: Error during TensorRT/CUDA cleanup: {e}")

                finally:
                    self.seg = None
                    print("TensorRT segmentor cleared")

            import gc
            gc.collect()
            print("Forced garbage collection completed")
            print("Processing thread ending naturally (full cleanup completed)")
            self.is_running = False

    def _process_frame(self, frame, current_video_time_seconds, inference_region, class_thresholds, frame_idx, frame_count):
        """Process a single frame with donut detection logic"""
        try:
            height, width = frame.shape[:2]
            
            lb, scale, left, top = letterbox(frame, size=self.config["canvas_size"], pad_val=114)

            boxes, labels, scores, _ = self.seg.infer_fast(lb, measure_gpu=False)

            if len(boxes) > 0:
                filtered_boxes, filtered_labels, filtered_scores, _, keep_indices = filter_by_class_thresholds(
                    boxes, labels, scores, np.array([]), class_thresholds)
            else:
                filtered_boxes, filtered_labels, filtered_scores, keep_indices = boxes, labels, scores, np.array([])

            if len(filtered_boxes) == 0:
                draw_inference_region(frame, inference_region)
                return frame

            boxes_img = unletterbox_boxes(filtered_boxes, scale, left, top, width, height)

            if len(keep_indices) > 0:
                masks_fp, _ = self.seg.copy_masks(keep_indices.astype(np.int64))
            else:
                masks_fp = np.array([])

            if masks_fp.size > 0:
                if self.config["mask_threshold"] > 0:
                    masks_u8 = (masks_fp > self.config["mask_threshold"]).astype(np.uint8) * 255
                else:
                    masks_u8 = masks_fp.astype(np.uint8)
                
                if masks_u8.size > 0:
                    masks_img, _, _ = unletterbox_masks_with_timing(
                        masks_u8, scale, left, top, width, height)
                else:
                    masks_img = np.array([])
            else:
                masks_img = np.array([])

            region_boxes, region_labels, region_scores, region_masks = filter_detections_in_region(
                boxes_img, filtered_labels, filtered_scores, masks_img, inference_region)

            draw_inference_region(frame, inference_region)

            if len(region_boxes) > 0:
                if region_masks.size > 0:
                    frame = colorize_doughnut_masks(frame, region_masks, region_boxes, region_labels, self.config["alpha"])

                doughnut_mask = (region_labels == 0) | (region_labels == 1)
                hole_mask = region_labels == 2

                doughnut_boxes = region_boxes[doughnut_mask]
                doughnut_labels = region_labels[doughnut_mask]
                hole_boxes = region_boxes[hole_mask]

                if len(doughnut_boxes) > 0:
                    numbered_doughnuts = draw_doughnut_numbers(frame, region_boxes, region_labels, inference_region)

                    if numbered_doughnuts and len(hole_boxes) > 0:
                        matches = match_holes_to_doughnuts(doughnut_boxes, doughnut_labels, hole_boxes)

                        for doughnut_idx, hole_idx in matches:
                            doughnut_box = doughnut_boxes[doughnut_idx]
                            hole_box = hole_boxes[hole_idx]
                            draw_measurement_lines(frame, doughnut_box, hole_box)

                        should_record, row_number = self.row_tracker.should_process_row(current_video_time_seconds)

                        if should_record:
                            self.row_counter = row_number
                            print(f"Recording Row {row_number} at video timestamp {current_video_time_seconds:.2f}s with {len(numbered_doughnuts)} doughnuts")
                            
                            self._process_row_data(numbered_doughnuts, doughnut_boxes, doughnut_labels, 
                                                 hole_boxes, matches, region_boxes, row_number)

            return frame

        except Exception as e:
            print(f"Error in frame processing: {e}")
            return frame

    def _process_row_data(self, numbered_doughnuts, doughnut_boxes, doughnut_labels, hole_boxes, matches, region_boxes, row_number):
        """Process and record row data for CSV and MQTT"""
        try:
            doughnut_hole_groups = {}
            for doughnut_idx, hole_idx in matches:
                if doughnut_idx not in doughnut_hole_groups:
                    doughnut_hole_groups[doughnut_idx] = []
                doughnut_hole_groups[doughnut_idx].append(hole_idx)

            row_data = []

            for num, orig_idx, _ in numbered_doughnuts:
                doughnut_found = False
                for doughnut_idx, (doughnut_box, doughnut_label) in enumerate(zip(doughnut_boxes, doughnut_labels)):
                    if orig_idx < len(region_boxes) and np.allclose(region_boxes[orig_idx], doughnut_box):
                        doughnut_found = True

                        if doughnut_idx in doughnut_hole_groups:
                            hole_indices = doughnut_hole_groups[doughnut_idx]

                            total_measurement_diff = 0
                            hole_count = len(hole_indices)

                            for hole_idx in hole_indices:
                                hole_box = hole_boxes[hole_idx]
                                top_distance, bottom_distance = calculate_measurements(doughnut_box, hole_box)
                                total_measurement_diff += abs(top_distance - bottom_distance)

                            avg_measurement_diff = total_measurement_diff / hole_count if hole_count > 0 else 0
                            measurement_value = round(avg_measurement_diff, 2)
                        else:
                            measurement_value = "null"

                        status = "good" if doughnut_label == 1 else "bad"
                        
                        self._add_csv_data(row_number, num, status, measurement_value)
                        
                        row_data.append({
                            'doughnut_number': num,
                            'status': status,
                            'measurement_difference': measurement_value
                        })
                        break

                if not doughnut_found:
                    print(f"Warning: Could not find doughnut {num} in filtered results")

            if self.mqtt_client and row_data:
                mqtt_payload = {
                    'row_number': row_number,
                    'doughnuts': row_data,
                    'timestamp': time.time(),
                    'total_doughnuts': len(row_data)
                }
                
                payload_json = json.dumps(mqtt_payload)
                self.mqtt_client.publish(self.config["mqtt_topic"], payload_json)
                print(f"Published MQTT data for row {row_number}: {len(row_data)} doughnuts")

        except Exception as e:
            print(f"Error processing row data: {e}")

    def _add_fps_overlay(self, vis, width):
        fps_text = f"FPS: {self.current_fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(
            vis,
            (width - text_size[0] - 20, 100),
            (width - 5, text_size[1] + 115),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            vis,
            fps_text,
            (width - text_size[0] - 15, text_size[1] + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _add_info_overlay(self, vis, current_video_time_seconds, frame_idx, frame_count, width):
        cv2.putText(vis, f"Time: {current_video_time_seconds:.1f}s", (width - 150, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Frame: {frame_idx+1}/{frame_count}", (width - 150, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.row_tracker:
            cv2.putText(vis, f"Row: {self.row_tracker.current_row_number}", (width - 150, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _emit_frame(self, frame, metadata):
        if not self.socketio or not self.websocket_mode:
            return

        current_time = time.time()
        if (current_time - self.last_websocket_frame_time < self.websocket_frame_interval):
            return

        self.last_websocket_frame_time = current_time

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        data = {"frame": frame_base64, "metadata": metadata}

        self.socketio.emit("stream_frame", data, namespace="/ws")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "current_fps": self.current_fps,
            "row_count": self.row_counter,
            "config": self.config,
            "initialization_error": self.initialization_error,
            "is_stuck": self.is_stuck(),
            "last_activity": time.time() - self.last_activity_time,
        }


class FPSCounter:
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.last_time = time.perf_counter()

    def update(self):
        current_time = time.perf_counter()
        self.times.append(current_time - self.last_time)
        self.last_time = current_time

    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        return len(self.times) / sum(self.times)


class TRTSegmentor:
    def __init__(self, engine_path, verbose=False):
        print(f"Initializing TRTSegmentor with engine: {engine_path}")

        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.ctx.push()

        try:
            self.trt10 = int(trt.__version__.split('.')[0]) >= 10
            logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)

            t0 = time.perf_counter()
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            t1 = time.perf_counter()
            self.model_load_s = (t1 - t0)

            names = ([self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                     if self.trt10 else
                     [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)])

            def has(n):
                return any(n == x for x in names)

            def find(sub):
                for n in names:
                    if sub in n:
                        return n
                return None

            self.name_in = "raw_input" if has("raw_input") else find("raw_input")
            self.name_dets = "dets" if has("dets") else find("dets")
            self.name_labels = "labels" if has("labels") else find("label")
            self.name_masks = "masks" if has("masks") else find("mask")
            if any(x is None for x in [self.name_in, self.name_dets, self.name_labels, self.name_masks]):
                raise RuntimeError(f"Could not find expected tensors. Found: {names}")

            def nptype_of(name):
                if self.trt10:
                    return trt.nptype(self.engine.get_tensor_dtype(name))
                else:
                    idx = self.engine.get_binding_index(name)
                    return trt.nptype(self.engine.get_binding_dtype(idx))

            self.dtype_in = nptype_of(self.name_in)
            self.dtype_dets = nptype_of(self.name_dets)
            self.dtype_labels = nptype_of(self.name_labels)
            self.dtype_masks = nptype_of(self.name_masks)

            def eng_shape(name):
                return tuple(self.engine.get_tensor_shape(name)) if self.trt10 \
                       else tuple(self.engine.get_binding_shape(self.engine.get_binding_index(name)))

            def ctx_shape(name):
                return tuple(self.context.get_tensor_shape(name)) if self.trt10 \
                       else tuple(self.context.get_binding_shape(self.engine.get_binding_index(name)))

            shp_in_engine = eng_shape(self.name_in)

            if self.trt10:
                shp_in_rt = shp_in_engine
                if -1 in shp_in_rt:
                    H, W = (shp_in_rt[-3], shp_in_rt[-2]) if shp_in_rt[-1] == 3 else (shp_in_rt[-2], shp_in_rt[-1])
                    shp_in_rt = (1, H if H > 0 else DEFAULT_CANVAS, W if W > 0 else DEFAULT_CANVAS, 3)
                    self.context.set_input_shape(self.name_in, shp_in_rt)
            else:
                b = self.engine.get_binding_index(self.name_in)
                shp_in_rt = shp_in_engine
                if -1 in shp_in_engine:
                    if self.engine.num_optimization_profiles > 0:
                        self.context.active_optimization_profile = 0
                    H, W = (shp_in_engine[-3], shp_in_engine[-2]) if shp_in_engine[-1] == 3 else (shp_in_engine[-2], shp_in_engine[-1])
                    shp_in_rt = (1, H if H > 0 else DEFAULT_CANVAS, W if W > 0 else DEFAULT_CANVAS, 3)
                    self.context.set_binding_shape(b, shp_in_rt)

            shp_in_rt_now = ctx_shape(self.name_in)
            shp_dets_ctx = ctx_shape(self.name_dets)
            shp_dets_eng = eng_shape(self.name_dets)
            shp_labels_ctx = ctx_shape(self.name_labels)
            shp_labels_eng = eng_shape(self.name_labels)
            shp_masks_ctx = ctx_shape(self.name_masks)
            shp_masks_eng = eng_shape(self.name_masks)

            def finalize(ctx_shp, eng_shp, kind):
                def ok(s):
                    return s and all((d is not None and d > 0) for d in s)

                if ok(ctx_shp):
                    return tuple(ctx_shp)
                if ok(eng_shp):
                    return tuple(eng_shp)
                if kind == "in":
                    H = eng_shp[-3] if eng_shp and eng_shp[-3] > 0 else DEFAULT_CANVAS
                    W = eng_shp[-2] if eng_shp and eng_shp[-2] > 0 else DEFAULT_CANVAS
                    return (1, H, W, 3)
                if kind == "dets":
                    N = eng_shp[1] if eng_shp and len(eng_shp) > 1 and eng_shp[1] > 0 else DEFAULT_TOPK
                    return (1, N, 5)
                if kind == "labels":
                    N = shp_dets_eng[1] if shp_dets_eng and len(shp_dets_eng) > 1 and shp_dets_eng[1] > 0 else DEFAULT_TOPK
                    return (1, N)
                if kind == "masks":
                    N = shp_dets_eng[1] if shp_dets_eng and len(shp_dets_eng) > 1 and shp_dets_eng[1] > 0 else DEFAULT_TOPK
                    Hm = shp_in_rt_now[-3] if shp_in_rt_now and len(shp_in_rt_now) == 4 else DEFAULT_CANVAS
                    Wm = shp_in_rt_now[-2] if shp_in_rt_now and len(shp_in_rt_now) == 4 else DEFAULT_CANVAS
                    return (1, N, Hm, Wm)
                raise ValueError(kind)

            shp_in_final = finalize(shp_in_rt_now, shp_in_engine, "in")
            shp_dets_final = finalize(shp_dets_ctx, shp_dets_eng, "dets")
            shp_labels_final = finalize(shp_labels_ctx, shp_labels_eng, "labels")
            shp_masks_final = finalize(shp_masks_ctx, shp_masks_eng, "masks")

            self.max_det = shp_dets_final[1]
            self.mask_hw = shp_masks_final[-2:]

            self.dev_ptr, self.host_buf, self.pinned_flag = {}, {}, {}

            def allocate_host(name, shape, dtype, force_pinned=False):
                numel = int(np.prod(shape))
                nbytes = numel * np.dtype(dtype).itemsize
                use_pinned = force_pinned or (nbytes >= PINNED_THRESHOLD_BYTES)
                buf = (cuda.pagelocked_empty(numel, dtype) if use_pinned else np.empty(numel, dtype=dtype))
                self.host_buf[name] = buf
                self.pinned_flag[name] = use_pinned
                self.dev_ptr[name] = cuda.mem_alloc(nbytes)

            def allocate_dev_only(name, shape, dtype):
                numel = int(np.prod(shape))
                nbytes = numel * np.dtype(dtype).itemsize
                self.dev_ptr[name] = cuda.mem_alloc(nbytes)

            allocate_host(self.name_in, shp_in_final, self.dtype_in)
            allocate_host(self.name_dets, shp_dets_final, self.dtype_dets, force_pinned=True)
            allocate_host(self.name_labels, shp_labels_final, self.dtype_labels, force_pinned=True)
            allocate_dev_only(self.name_masks, shp_masks_final, self.dtype_masks)

            self.stream = cuda.Stream()
            self.start_evt = cuda.Event()
            self.end_evt = cuda.Event()

            self.output_boxes = np.empty((self.max_det, 4), dtype=np.float32)
            self.output_labels = np.empty(self.max_det, dtype=np.int32)
            self.output_scores = np.empty(self.max_det, dtype=np.float32)

            self.mask_host_capacity = 0
            self.mask_host_buf = None

            if self.trt10:
                for n, d in self.dev_ptr.items():
                    self.context.set_tensor_address(n, int(d))
            else:
                self.bindings_order = [None] * self.engine.num_bindings
                for i in range(self.engine.num_bindings):
                    n = self.engine.get_binding_name(i)
                    self.bindings_order[i] = int(self.dev_ptr[n])

            print("TRTSegmentor initialization complete")

        finally:
            pass

    def infer_fast(self, lb_img_uint8, measure_gpu=False):
        """Optimized inference with minimal sync points"""
        np.copyto(self.host_buf[self.name_in], lb_img_uint8.ravel())
        if self.pinned_flag[self.name_in]:
            cuda.memcpy_htod_async(self.dev_ptr[self.name_in], self.host_buf[self.name_in], self.stream)
        else:
            cuda.memcpy_htod(self.dev_ptr[self.name_in], self.host_buf[self.name_in])

        if measure_gpu:
            self.stream.synchronize()
            self.start_evt.record(self.stream)

        if self.trt10:
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(self.bindings_order, self.stream.handle)

        if measure_gpu:
            self.end_evt.record(self.stream)
            self.stream.synchronize()
            gpu_s = self.start_evt.time_till(self.end_evt) / 1e3
        else:
            gpu_s = 0.0

        if not measure_gpu:
            cuda.memcpy_dtoh_async(self.host_buf[self.name_dets], self.dev_ptr[self.name_dets], self.stream)
            cuda.memcpy_dtoh_async(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels], self.stream)
            self.stream.synchronize()
        else:
            cuda.memcpy_dtoh(self.host_buf[self.name_dets], self.dev_ptr[self.name_dets])
            cuda.memcpy_dtoh(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels])

        dets_raw = self.host_buf[self.name_dets].view().reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].view().reshape(1, self.max_det)[0]

        np.copyto(self.output_boxes, dets_raw[:, :4])
        np.copyto(self.output_scores, dets_raw[:, 4])
        np.copyto(self.output_labels, labels_raw.astype(np.int32))

        return self.output_boxes, self.output_labels, self.output_scores, gpu_s

    def _ensure_mask_host(self, need_count):
        Hm, Wm = self.mask_hw
        if self.mask_host_buf is not None and need_count <= self.mask_host_capacity:
            return
        new_cap = max(need_count, max(1, self.mask_host_capacity * 2))
        self.mask_host_buf = cuda.pagelocked_empty((new_cap, Hm, Wm), dtype=self.dtype_masks)
        self.mask_host_capacity = new_cap

    def copy_masks(self, indices):
        """Pinned + async D2H for kept slices only."""
        Hm, Wm = self.mask_hw
        dtype = self.dtype_masks
        itemsize = np.dtype(dtype).itemsize
        slice_bytes = Hm * Wm * itemsize

        if len(indices) == 0:
            return np.empty((0, Hm, Wm), dtype=dtype), 0.0

        self._ensure_mask_host(len(indices))
        mask_stream = cuda.Stream()
        t0 = time.perf_counter()
        base_addr = int(self.dev_ptr[self.name_masks])
        for k, idx in enumerate(indices):
            src = base_addr + int(idx) * slice_bytes
            dst = self.mask_host_buf[k].ravel()
            cuda.memcpy_dtoh_async(dst, src, mask_stream)
        mask_stream.synchronize()
        t1 = time.perf_counter()
        return np.array(self.mask_host_buf[:len(indices)]), (t1 - t0)

    def cleanup(self):
        """Clean up all CUDA resources with timeout protection"""
        print("Cleaning up TRTSegmentor CUDA resources...")

        cleanup_start_time = time.time()
        cleanup_timeout = 5.0

        try:
            try:
                if hasattr(self, "stream") and self.stream:
                    print("Synchronizing CUDA stream...")
                    self.stream.synchronize()
                    print("CUDA stream synchronized")
            except Exception as e:
                print(f"Warning: Could not synchronize CUDA stream: {e}")

            if time.time() - cleanup_start_time > cleanup_timeout:
                print("Cleanup timeout reached during stream sync - aborting")
                return

            print("Freeing device memory...")
            freed_count = 0
            for name, ptr in list(self.dev_ptr.items()):
                try:
                    if ptr:
                        ptr.free()
                        freed_count += 1
                except Exception as e:
                    print(f"Warning: Error freeing device memory for {name}: {e}")

                if time.time() - cleanup_start_time > cleanup_timeout:
                    print(f"Cleanup timeout reached - freed {freed_count}/{len(self.dev_ptr)} buffers")
                    break

            print(f"Freed {freed_count} device memory buffers")

            self.host_buf.clear()
            self.dev_ptr.clear()
            self.pinned_flag.clear()

            self.mask_host_buf = None
            self.mask_host_capacity = 0

            print("CUDA resources cleanup attempted")

        except Exception as e:
            print(f"Warning: Error during CUDA resource cleanup: {e}")

        finally:
            print("Detaching CUDA context...")
            try:
                if hasattr(self, "ctx") and self.ctx:
                    self.ctx.pop()
                    self.ctx.detach()
                    print("CUDA context detached successfully")
            except Exception as e:
                print(f"Warning: TensorRT context cleanup error (IGNORED): {e}")
                print("This TensorRT cleanup warning can be safely ignored")

            try:
                time.sleep(0.1)
            except:
                pass

            print("TensorRT cleanup completed (with warnings ignored)")