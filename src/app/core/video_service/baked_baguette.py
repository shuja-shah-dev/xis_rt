import argparse
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque
import json
import os
import threading
import queue
import base64
import csv

PINNED_THRESHOLD_BYTES = 32 * 1024 * 1024
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640
CV_CN_MAX_SAFE = 256


def apply_nms(boxes, scores, labels, nms_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
    boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=nms_threshold,
    )

    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], scores[indices], labels[indices]
    else:
        return np.array([]), np.array([]), np.array([])


class VideoInferenceService_baked:
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
        self.tray_counter = 0
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

    def set_socketio(self, socketio):
        self.socketio = socketio

    def set_mqtt(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def initialize_from_config(self, config):
        self.config = {
            "engine_path": config.get("engine_path", ""),
            "video_path": config.get("video_path", ""),
            "score_threshold": config.get("score_threshold", 0.4),
            "score_class0": config.get("score_class0", None),
            "score_class1": config.get("score_class1", None),
            "score_class2": config.get("score_class2", None),
            "score_class3": config.get("score_class3", None),
            "nms_threshold": config.get("nms_threshold", 0.5),
            "mask_threshold": config.get("mask_threshold", 0.4),
            "canvas_size": config.get("canvas_size", 640),
            "alpha": config.get("alpha", 0.3),
            "min_inference_frames": config.get("min_inference_frames", 30),
            "target_fps": config.get("target_fps", 30.0),
            "mqtt_topic": config.get("mqtt_topic", "detection/results"),
            "json_output_dir": config.get("json_output_dir", None),
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

    def _add_csv_data(self, tray_number, defected_count, good_count, acceptable_count):
        """Add tray data to CSV collection with new column structure"""
        csv_row = {
            "tray_number": f"Tray {tray_number}",
            "good": good_count,
            "acceptable": acceptable_count,
            "defected": defected_count,
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
            csv_filename = f"baguette_analysis_{timestamp_str}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)

            with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["tray_number", "good", "acceptable", "defected"]
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
                    print(
                        f"MAJOR FREEZE DETECTED - EMERGENCY SHUTDOWN after {time_since_activity:.1f}s!"
                    )
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
        self.tray_counter = 0
        self.inference_active = False
        self.initialization_error = None
        self.last_activity_time = time.time()

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

    def is_inference_timestamp(self, timestamp):
        """Check if timestamp matches the pattern: 2.2, 5.2, 8.2, etc."""
        target_timestamps = []
        t = 2.2
        while t <= timestamp + 1.0:
            target_timestamps.append(t)
            t += 3.0

        for target in target_timestamps:
            if abs(timestamp - target) <= 0.1:
                return target
        return None

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
            frame_idx = 0
            fps_counter = FPSCounter()

            inference_active = False
            inference_frame_count = 0
            last_inference_timestamp = -1
            min_inference_frames = self.config["min_inference_frames"]

            print("Video processing loop started")
            print("Inference runs at 2.2s, 5.2s, 8.2s... (increment by 3)")

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

                    current_timestamp = frame_idx / fps if fps > 0 else 0

                    target_timestamp = self.is_inference_timestamp(current_timestamp)
                    should_start_inference = (
                        target_timestamp is not None
                        and target_timestamp != last_inference_timestamp
                        and not inference_active
                    )

                    if should_start_inference:
                        inference_active = True
                        inference_frame_count = 0
                        last_inference_timestamp = target_timestamp
                        self.tray_counter += 1
                        print(
                            f"Starting inference for Tray {self.tray_counter} at timestamp {current_timestamp:.1f}s"
                        )

                    if inference_active:
                        inference_frame_count += 1
                        if inference_frame_count >= min_inference_frames:
                            inference_active = False
                            print(
                                f"Completed inference period ({inference_frame_count} frames)"
                            )

                    if inference_active:
                        vis, detection_count = self._process_inference_frame(
                            frame, inference_frame_count, current_timestamp
                        )
                    else:
                        vis = frame.copy()
                        detection_count = 0

                        next_target = 2.2
                        while next_target <= current_timestamp:
                            next_target += 3.0

                        cv2.putText(
                            vis,
                            f"INFERENCE OFF (Timestamp: {current_timestamp:.1f}s)",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        cv2.putText(
                            vis,
                            f"Next inference: {next_target:.1f}s",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

                    fps_counter.update()
                    self.current_fps = fps_counter.get_fps()
                    self.inference_active = inference_active

                    # self._add_fps_overlay(vis)
                    cv2.putText(
                        vis,
                        f"Frame: {frame_idx} | Time: {current_timestamp:.1f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    self._emit_frame(
                        vis,
                        {
                            "timestamp": current_timestamp,
                            "inference_active": inference_active,
                            "fps": self.current_fps,
                            "frame_idx": frame_idx,
                            "tray_number": self.tray_counter,
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
                                print(
                                    f"No more contexts to pop (popped {context_count} total)"
                                )
                                break

                        try:
                            self.seg.ctx.detach()
                            print("CUDA context detached")
                        except Exception as detach_error:
                            print(
                                f"Context detach not needed or failed: {detach_error}"
                            )

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

    def _process_inference_frame(self, frame, inference_frame_count, current_timestamp):
        try:
            lb, scale, left, top = letterbox(
                frame, size=self.config["canvas_size"], pad_val=114
            )

            boxes, labels, scores, _ = self.seg.infer_fast(lb, measure_gpu=False)

            keep = np.zeros(len(scores), dtype=bool)
            for i in range(len(scores)):
                label_val = int(labels[i])
                if label_val == 0 and self.config["score_class0"] is not None:
                    threshold = self.config["score_class0"]
                elif label_val == 1 and self.config["score_class1"] is not None:
                    threshold = self.config["score_class1"]
                elif label_val == 2 and self.config["score_class2"] is not None:
                    threshold = self.config["score_class2"]
                elif label_val == 3 and self.config["score_class3"] is not None:
                    threshold = self.config["score_class3"]
                else:
                    threshold = self.config["score_threshold"]

                keep[i] = scores[i] >= threshold

            boxes_filtered = boxes[keep]
            labels_filtered = labels[keep]
            scores_filtered = scores[keep]

            if len(boxes_filtered) > 0:
                boxes_nms, scores_nms, labels_nms = apply_nms(
                    boxes_filtered,
                    scores_filtered,
                    labels_filtered,
                    nms_threshold=self.config["nms_threshold"],
                )
            else:
                boxes_nms, scores_nms, labels_nms = (
                    boxes_filtered,
                    scores_filtered,
                    labels_filtered,
                )

            if len(boxes_nms) > 0:
                keep_idx = []
                original_indices = np.where(keep)[0]

                for nms_box, nms_score, nms_label in zip(
                    boxes_nms, scores_nms, labels_nms
                ):
                    for j, orig_idx in enumerate(original_indices):
                        if (
                            np.allclose(boxes_filtered[j], nms_box, atol=1e-5)
                            and np.isclose(scores_filtered[j], nms_score, atol=1e-5)
                            and labels_filtered[j] == nms_label
                        ):
                            keep_idx.append(orig_idx)
                            break

                keep_idx = np.array(keep_idx, dtype=np.int64)
            else:
                keep_idx = np.array([], dtype=np.int64)

            boxes_img = unletterbox_boxes(
                boxes_nms, scale, left, top, frame.shape[1], frame.shape[0]
            )

            tray_indices = [i for i, lab in enumerate(labels_nms) if int(lab) == 3]

            if len(tray_indices) > 0:
                tray_bbox = boxes_img[tray_indices[0]]
                tray_x1, tray_y1, tray_x2, tray_y2 = tray_bbox

                within_tray_mask = []
                for i, (box, lab) in enumerate(zip(boxes_img, labels_nms)):
                    if int(lab) == 3:
                        within_tray_mask.append(True)
                    else:
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = box
                        tolerance = 10
                        within_tray = (
                            bbox_x1 >= (tray_x1 - tolerance)
                            and bbox_y1 >= (tray_y1 - tolerance)
                            and bbox_x2 <= (tray_x2 + tolerance)
                            and bbox_y2 <= (tray_y2 + tolerance)
                        )
                        within_tray_mask.append(within_tray)

                within_tray_mask = np.array(within_tray_mask)

                boxes_img = boxes_img[within_tray_mask]
                labels_nms = labels_nms[within_tray_mask]
                keep_idx = keep_idx[within_tray_mask]

            if keep_idx.size > 0:
                masks_fp, _ = self.seg.copy_masks(keep_idx.astype(np.int64))
            else:
                masks_fp = np.empty((0, *self.seg.mask_hw), dtype=self.seg.dtype_masks)

            masks_u8, _ = self._to_u8_fast(masks_fp)

            should_publish = inference_frame_count == 20

            vis, measurement_data = self._measure_and_visualize_baguettes_fast(
                frame,
                masks_u8,
                boxes_img,
                labels_nms,
                scale,
                left,
                top,
                should_publish,
                current_timestamp,
            )

            if should_publish:
                class_0_count = measurement_data["defected"]
                class_1_count = measurement_data["good"]
                class_2_count = measurement_data["acceptable"]

                self._add_csv_data(
                    self.tray_counter, class_0_count, class_1_count, class_2_count
                )

                if self.mqtt_client:
                    payload = json.dumps(
                        {
                            "tray_number": self.tray_counter,
                            "good": class_1_count,
                            "acceptable": class_2_count,
                            "defected": class_0_count,
                            "total_detected": measurement_data["total_detected"],
                            "timestamp": current_timestamp,
                            "processing_time": time.time(),
                        }
                    )
                    self.mqtt_client.publish(self.config["mqtt_topic"], payload)

            cv2.putText(
                vis,
                f"INFERENCE ON (Frame {inference_frame_count}/{self.config['min_inference_frames']})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            return vis, len(boxes_img)

        except Exception as e:
            print(f"Error in inference frame: {e}")
            return frame.copy(), 0

    def _measure_and_visualize_baguettes_fast(
        self,
        base_bgr,
        masks_lb_u8,
        boxes_img,
        labels,
        scale,
        left,
        top,
        should_publish,
        current_timestamp,
    ):
        """Measurement and visualization for new factory setup with 4 classes and vertical baguettes"""
        vis = base_bgr.copy()
        if boxes_img.size == 0:
            return vis, {"defected": 0, "good": 0, "acceptable": 0, "total_detected": 0}

        H_img, W_img = vis.shape[:2]

        tray_indices = [i for i, lab in enumerate(labels) if int(lab) == 3]
        tray_bbox = None
        if tray_indices:
            tray_bbox = boxes_img[tray_indices[0]]

        baguette_indices = [i for i, lab in enumerate(labels) if int(lab) in [0, 1, 2]]

        objects = []
        for i in baguette_indices:
            box = boxes_img[i]
            lab = labels[i]
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width_px = x2 - x1
            length_px = y2 - y1

            objects.append(
                {
                    "index": i,
                    "bbox": box,
                    "label": lab,
                    "center_x": center_x,
                    "center_y": center_y,
                    "top_edge": y1,
                    "bottom_edge": y2,
                    "width_px": width_px,
                    "length_px": length_px,
                }
            )

        if should_publish and current_timestamp is not None:
            print(
                f"\n[Timestamp {current_timestamp:.1f}s - Frame 20 of inference period] BAGUETTE MEASUREMENT ANALYSIS"
            )
            print("=" * 60)

            class_0_count = sum(1 for obj in objects if int(obj["label"]) == 0)
            class_1_count = sum(1 for obj in objects if int(obj["label"]) == 1)
            class_2_count = sum(1 for obj in objects if int(obj["label"]) == 2)

            print("BAGUETTE QUALITY SUMMARY:")
            print(f"   Good baguettes (Class 1):       {class_1_count}")
            print(f"   Acceptable baguettes (Class 2): {class_2_count}")
            print(f"   Defected baguettes (Class 0):   {class_0_count}")
            print(f"   Total detected:                 {len(objects)}")
            if tray_bbox is not None:
                print(f"   Tray detected:                  Yes")
            else:
                print(f"   Tray detected:                  No")

            if self.config.get("json_output_dir"):
                self._save_json_data(
                    objects,
                    tray_bbox,
                    self.tray_counter,
                    current_timestamp,
                    self.config["json_output_dir"],
                )

        if tray_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in tray_bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if tray_bbox is not None and len(objects) > 0:
            tray_y_center = (tray_bbox[1] + tray_bbox[3]) / 2
            row1_objects = [obj for obj in objects if obj["center_y"] < tray_y_center]
            row2_objects = [obj for obj in objects if obj["center_y"] >= tray_y_center]

            row1_objects.sort(key=lambda obj: obj["center_x"])
            row2_objects.sort(key=lambda obj: obj["center_x"])

            if should_publish:
                print(f"Row 1 (Top): {len(row1_objects)} baguettes")
                print(f"Row 2 (Bottom): {len(row2_objects)} baguettes")
        else:
            row1_objects = sorted(objects, key=lambda obj: obj["center_x"])
            row2_objects = []

        for obj in objects:
            i = obj["index"]
            box = obj["bbox"]
            lab = obj["label"]
            center_x, center_y = obj["center_x"], obj["center_y"]
            width_px, length_px = obj["width_px"], obj["length_px"]

            if int(lab) == 3:
                continue

            if i < len(masks_lb_u8):
                m = masks_lb_u8[i]
                x_lb, y_lb, w_lb, h_lb = cv2.boundingRect(m)
                if w_lb <= 0 or h_lb <= 0:
                    continue

                x1 = int(np.clip(round((x_lb - left) / scale), 0, W_img - 1))
                y1 = int(np.clip(round((y_lb - top) / scale), 0, H_img - 1))
                x2 = int(np.clip(round(((x_lb + w_lb) - left) / scale), 0, W_img))
                y2 = int(np.clip(round(((y_lb + h_lb) - top) / scale), 0, H_img))
                pw, ph = max(0, x2 - x1), max(0, y2 - y1)
                if pw == 0 or ph == 0:
                    continue

                patch = m[y_lb : y_lb + h_lb, x_lb : x_lb + w_lb]
                mroi = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_NEAREST)
                roi = vis[y1 : y1 + ph, x1 : x1 + pw]

                label_val = int(lab)

                if label_val == 0:
                    mask_color = (0, 0, 255)
                elif label_val == 1:
                    mask_color = (0, 255, 0)
                elif label_val == 2:
                    mask_color = (0, 255, 255)
                else:
                    mask_color = (128, 128, 128)

                color_roi = np.empty_like(roi)
                color_roi[:] = mask_color
                blended = cv2.addWeighted(
                    roi,
                    1.0 - self.config["alpha"],
                    color_roi,
                    self.config["alpha"],
                    0.0,
                )
                cv2.copyTo(blended, mroi, roi)

            x1b, y1b, x2b, y2b = [int(v) for v in box]
            center_x_int, center_y_int = int(center_x), int(center_y)

            cv2.line(vis, (center_x_int, y1b), (center_x_int, y2b), (255, 0, 0), 1)
            cv2.line(vis, (x1b, center_y_int), (x2b, center_y_int), (0, 0, 0), 1)

        if should_publish:
            for row_num, row_objects in enumerate([row1_objects, row2_objects], 1):
                if not row_objects:
                    continue

                print(f"\nROW {row_num} MEASUREMENTS")
                print("-" * 40)

                for obj_num, obj in enumerate(row_objects, 1):
                    label_val = int(obj["label"])
                    if label_val == 0:
                        color_name = "Red (Defected)"
                    elif label_val == 1:
                        color_name = "Green (Good)"
                    elif label_val == 2:
                        color_name = "Yellow (Acceptable)"
                    else:
                        color_name = "Other"

                    print(
                        f"Baguette {obj_num:2d} ({color_name}): Length={obj['width_px']:6.1f}px, Width={obj['length_px']:6.1f}px"
                    )

            print("\n" + "=" * 60)
            print("FINAL SUMMARY:")
            total_baguettes = len(objects)
            class_0_count = sum(1 for obj in objects if int(obj["label"]) == 0)
            class_1_count = sum(1 for obj in objects if int(obj["label"]) == 1)
            class_2_count = sum(1 for obj in objects if int(obj["label"]) == 2)

            if total_baguettes > 0:
                good_rate = (class_1_count / total_baguettes) * 100
                acceptable_rate = (class_2_count / total_baguettes) * 100
                defect_rate = (class_0_count / total_baguettes) * 100
                print(
                    f"   Good Rate:       {good_rate:.1f}% ({class_1_count}/{total_baguettes})"
                )
                print(
                    f"   Acceptable Rate: {acceptable_rate:.1f}% ({class_2_count}/{total_baguettes})"
                )
                print(
                    f"   Defect Rate:     {defect_rate:.1f}% ({class_0_count}/{total_baguettes})"
                )
            else:
                print("   No baguettes detected")
            print("=" * 60)

        measurement_data = {
            "defected": sum(1 for obj in objects if int(obj["label"]) == 0),
            "good": sum(1 for obj in objects if int(obj["label"]) == 1),
            "acceptable": sum(1 for obj in objects if int(obj["label"]) == 2),
            "total_detected": len(objects),
        }

        return vis, measurement_data

    def _save_json_data(
        self, objects, tray_bbox, tray_number, timestamp, json_output_dir
    ):
        """Save detailed baguette measurement data to JSON file"""
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)

        class_0_count = sum(1 for obj in objects if int(obj["label"]) == 0)
        class_1_count = sum(1 for obj in objects if int(obj["label"]) == 1)
        class_2_count = sum(1 for obj in objects if int(obj["label"]) == 2)

        if tray_bbox is not None:
            tray_y_center = (tray_bbox[1] + tray_bbox[3]) / 2
            row1_objects = [obj for obj in objects if obj["center_y"] < tray_y_center]
            row2_objects = [obj for obj in objects if obj["center_y"] >= tray_y_center]

            row1_objects.sort(key=lambda obj: obj["center_x"])
            row2_objects.sort(key=lambda obj: obj["center_x"])
        else:
            row1_objects = objects
            row2_objects = []

        json_data = {
            "tray_number": f"Tray {tray_number}",
            "timestamp": f"{timestamp:.1f}s",
            "good": int(class_1_count),
            "acceptable": int(class_2_count),
            "defected": int(class_0_count),
            "total_detected": int(len(objects)),
            "row1_baguettes": len(row1_objects),
            "row2_baguettes": len(row2_objects),
            "baguettes": [],
        }

        all_sorted_objects = []

        for i, obj in enumerate(row1_objects):
            all_sorted_objects.append((f"row1_baguette_{i+1}", obj))

        for i, obj in enumerate(row2_objects):
            all_sorted_objects.append((f"row2_baguette_{i+1}", obj))

        for baguette_id, obj in all_sorted_objects:
            baguette_data = {
                baguette_id: {
                    "width": round(float(obj["length_px"]), 1),
                    "length": round(float(obj["width_px"]), 1),
                    "class": int(obj["label"]),
                    "classification": (
                        "defected"
                        if int(obj["label"]) == 0
                        else "good" if int(obj["label"]) == 1 else "acceptable"
                    ),
                    "position": {
                        "top_edge": round(float(obj["top_edge"]), 1),
                        "bottom_edge": round(float(obj["bottom_edge"]), 1),
                        "center_x": round(float(obj["center_x"]), 1),
                        "center_y": round(float(obj["center_y"]), 1),
                    },
                }
            }
            json_data["baguettes"].append(baguette_data)

        filename = f"tray_{tray_number:03d}_timestamp_{timestamp:.1f}s.json"
        filepath = os.path.join(json_output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"JSON data saved: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False

    def _add_fps_overlay(self, vis):
        height, width = vis.shape[:2]
        fps_text = f"FPS: {self.current_fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(
            vis,
            (width - text_size[0] - 20, 70),
            (width - 5, text_size[1] + 85),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            vis,
            fps_text,
            (width - text_size[0] - 15, text_size[1] + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    def _emit_frame(self, frame, metadata):
        if not self.socketio or not self.websocket_mode:
            return

        current_time = time.time()
        if (
            current_time - self.last_websocket_frame_time
            < self.websocket_frame_interval
        ):
            return

        self.last_websocket_frame_time = current_time

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        data = {"frame": frame_base64, "metadata": metadata}

        self.socketio.emit("stream_frame", data, namespace="/ws")

    def _to_u8_fast(self, m):
        if m.size == 0:
            return m.astype(np.uint8), False
        if self.config.get("engine_binary_masks", True):
            return m.astype(np.uint8, copy=False), False
        flat = m.ravel()
        sample = flat[:: max(1, flat.size // 2048)]
        if sample.max() <= 1.05:
            out = (m > self.config["mask_threshold"]).astype(np.uint8) * 255
            return out, True
        if np.all((sample == 0) | (np.abs(sample - 255.0) < 0.5)):
            return m.astype(np.uint8, copy=False), False
        out = (m > self.config["mask_threshold"]).astype(np.uint8) * 255
        return out, True

    def get_status(self):
        return {
            "is_running": self.is_running,
            "current_fps": self.current_fps,
            "tray_count": self.tray_counter,
            "inference_active": self.inference_active,
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


def letterbox(img, size=640, pad_val=114):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_val, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, scale, left, top


def unletterbox_boxes(boxes_xyxy, scale, left, top, out_w, out_h):
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    out = boxes_xyxy.astype(np.float32, copy=False).copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / scale, 0, out_w - 1)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top) / scale, 0, out_h - 1)
    return out


class TRTSegmentor:
    def __init__(self, engine_path, verbose=False):
        print(f"Initializing TRTSegmentor with engine: {engine_path}")

        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.ctx.push()

        try:
            self.trt10 = int(trt.__version__.split(".")[0]) >= 10
            logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.ERROR)

            t0 = time.perf_counter()
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            t1 = time.perf_counter()
            self.model_load_s = t1 - t0

            names = (
                [
                    self.engine.get_tensor_name(i)
                    for i in range(self.engine.num_io_tensors)
                ]
                if self.trt10
                else [
                    self.engine.get_binding_name(i)
                    for i in range(self.engine.num_bindings)
                ]
            )

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

            if any(
                x is None
                for x in [
                    self.name_in,
                    self.name_dets,
                    self.name_labels,
                    self.name_masks,
                ]
            ):
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
                return (
                    tuple(self.engine.get_tensor_shape(name))
                    if self.trt10
                    else tuple(
                        self.engine.get_binding_shape(
                            self.engine.get_binding_index(name)
                        )
                    )
                )

            def ctx_shape(name):
                return (
                    tuple(self.context.get_tensor_shape(name))
                    if self.trt10
                    else tuple(
                        self.context.get_binding_shape(
                            self.engine.get_binding_index(name)
                        )
                    )
                )

            shp_in_engine = eng_shape(self.name_in)

            if self.trt10:
                shp_in_rt = shp_in_engine
                if -1 in shp_in_rt:
                    H, W = (
                        (shp_in_rt[-3], shp_in_rt[-2])
                        if shp_in_rt[-1] == 3
                        else (shp_in_rt[-2], shp_in_rt[-1])
                    )
                    shp_in_rt = (
                        1,
                        H if H > 0 else DEFAULT_CANVAS,
                        W if W > 0 else DEFAULT_CANVAS,
                        3,
                    )
                    self.context.set_input_shape(self.name_in, shp_in_rt)
            else:
                b = self.engine.get_binding_index(self.name_in)
                shp_in_rt = shp_in_engine
                if -1 in shp_in_engine:
                    if self.engine.num_optimization_profiles > 0:
                        self.context.active_optimization_profile = 0
                    H, W = (
                        (shp_in_engine[-3], shp_in_engine[-2])
                        if shp_in_engine[-1] == 3
                        else (shp_in_engine[-2], shp_in_engine[-1])
                    )
                    shp_in_rt = (
                        1,
                        H if H > 0 else DEFAULT_CANVAS,
                        W if W > 0 else DEFAULT_CANVAS,
                        3,
                    )
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
                    N = (
                        eng_shp[1]
                        if eng_shp and len(eng_shp) > 1 and eng_shp[1] > 0
                        else DEFAULT_TOPK
                    )
                    return (1, N, 5)
                if kind == "labels":
                    N = (
                        shp_dets_eng[1]
                        if shp_dets_eng
                        and len(shp_dets_eng) > 1
                        and shp_dets_eng[1] > 0
                        else DEFAULT_TOPK
                    )
                    return (1, N)
                if kind == "masks":
                    N = (
                        shp_dets_eng[1]
                        if shp_dets_eng
                        and len(shp_dets_eng) > 1
                        and shp_dets_eng[1] > 0
                        else DEFAULT_TOPK
                    )
                    Hm = (
                        shp_in_rt_now[-3]
                        if shp_in_rt_now and len(shp_in_rt_now) == 4
                        else DEFAULT_CANVAS
                    )
                    Wm = (
                        shp_in_rt_now[-2]
                        if shp_in_rt_now and len(shp_in_rt_now) == 4
                        else DEFAULT_CANVAS
                    )
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
                buf = (
                    cuda.pagelocked_empty(numel, dtype)
                    if use_pinned
                    else np.empty(numel, dtype=dtype)
                )
                self.host_buf[name] = buf
                self.pinned_flag[name] = use_pinned
                self.dev_ptr[name] = cuda.mem_alloc(nbytes)

            def allocate_dev_only(name, shape, dtype):
                numel = int(np.prod(shape))
                nbytes = numel * np.dtype(dtype).itemsize
                self.dev_ptr[name] = cuda.mem_alloc(nbytes)

            allocate_host(self.name_in, shp_in_final, self.dtype_in)
            allocate_host(
                self.name_dets, shp_dets_final, self.dtype_dets, force_pinned=True
            )
            allocate_host(
                self.name_labels, shp_labels_final, self.dtype_labels, force_pinned=True
            )
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
        np.copyto(self.host_buf[self.name_in], lb_img_uint8.ravel())
        if self.pinned_flag[self.name_in]:
            cuda.memcpy_htod_async(
                self.dev_ptr[self.name_in], self.host_buf[self.name_in], self.stream
            )
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
            cuda.memcpy_dtoh_async(
                self.host_buf[self.name_dets], self.dev_ptr[self.name_dets], self.stream
            )
            cuda.memcpy_dtoh_async(
                self.host_buf[self.name_labels],
                self.dev_ptr[self.name_labels],
                self.stream,
            )
            self.stream.synchronize()
        else:
            cuda.memcpy_dtoh(
                self.host_buf[self.name_dets], self.dev_ptr[self.name_dets]
            )
            cuda.memcpy_dtoh(
                self.host_buf[self.name_labels], self.dev_ptr[self.name_labels]
            )

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
        self.mask_host_buf = cuda.pagelocked_empty(
            (new_cap, Hm, Wm), dtype=self.dtype_masks
        )
        self.mask_host_capacity = new_cap

    def copy_masks(self, indices):
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
        return np.array(self.mask_host_buf[: len(indices)]), (t1 - t0)

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
                    print(
                        f"Cleanup timeout reached - freed {freed_count}/{len(self.dev_ptr)} buffers"
                    )
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
