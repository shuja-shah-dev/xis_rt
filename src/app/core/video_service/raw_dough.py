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

PINNED_THRESHOLD_BYTES = 32 * 1024 * 1024
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640
CV_CN_MAX_SAFE = 256


class VideoInferenceService:
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
        self.seg = None  # Will be initialized in processing thread
        self.cap = None
        self.initialization_error = None
        self.force_stop_event = threading.Event()
        self.last_activity_time = time.time()
        self.processing_timeout = 30.0  # 30 second timeout

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
            "nms_threshold": config.get("nms_threshold", 0.5),
            "mask_threshold": config.get("mask_threshold", 0.4),
            "canvas_size": config.get("canvas_size", 640),
            "alpha": config.get("alpha", 0.3),
            "min_inference_frames": config.get("min_inference_frames", 30),
            "target_fps": config.get("target_fps", 30.0),
            "mqtt_topic": config.get("mqtt_topic", "detection/results"),
        }

        # Just validate that the engine path exists, don't initialize CUDA here
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

        # Test video file access
        test_cap = cv2.VideoCapture(self.config["video_path"])
        if not test_cap.isOpened():
            print(f"Could not open video file: {self.config['video_path']}")
            return False
        test_cap.release()

        self.is_running = True
        self.shutdown_event.clear()
        self.initialization_error = None

        # Start processing thread - CUDA initialization will happen there
        self.processing_thread = threading.Thread(
            target=self._process_video, daemon=True
        )
        self.processing_thread.start()

        # Give a moment for initialization and check if it succeeded
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
        if self.socketio:
            self.socketio.emit(
                "status",
                {"message": "Stream stopped", "streaming_active": False},
                namespace="/ws",
                to="stream",
            )

        if self.processing_thread and self.processing_thread.is_alive():
            print("Waiting for processing thread to finish...")
            # Give it 3 seconds to stop gracefully
            self.processing_thread.join(timeout=3.0)

            if self.processing_thread.is_alive():
                print("WARNING: Processing thread did not terminate gracefully")

        # Reset to initial state instead of manual cleanup
        self.reset_to_initial_state()
        print("Video processing stopped and service reset")
        return True

    def reset_to_initial_state(self):
        """Reset the service back to initial state - avoids TensorRT cleanup issues"""
        print("Resetting VideoInferenceService to initial state...")

        # Release video capture safely
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
                print("Video capture released")
        except Exception as e:
            print(f"Warning: Error releasing video capture: {e}")

        # Reset all instance variables to initial state
        self.is_running = False
        self.processing_thread = None
        self.current_fps = 0
        self.last_websocket_frame_time = 0
        self.tray_counter = 0
        self.inference_active = False
        self.initialization_error = None
        self.last_activity_time = time.time()

        # Clear events
        self.shutdown_event.clear()
        self.force_stop_event.clear()

        # Clear frame queue
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass

        # Most importantly - set seg to None and let the thread handle cleanup naturally
        # This avoids the problematic TensorRT destructor call
        self.seg = None

        print("Service reset to initial state completed")

    def force_stop(self):
        """Emergency stop function"""
        print("EMERGENCY STOP: Force stopping video processing...")
        self.is_running = False
        self.shutdown_event.set()
        self.force_stop_event.set()

        # Just reset instead of manual cleanup
        self.reset_to_initial_state()

        if self.socketio:
            self.socketio.emit(
                "video_status",
                {"status": "force_stopped", "reason": "emergency_stop"},
                namespace="/ws",
            )

        return True

    def force_cleanup(self):
        """Force cleanup of resources - now just calls reset"""
        self.reset_to_initial_state()

    def _process_video(self):
        """Main video processing loop - runs in separate thread"""
        print("Starting video processing thread...")

        # Initialize CUDA/TensorRT in this thread
        if not self._initialize_engine_in_thread():
            print("Failed to initialize engine in processing thread")
            self.is_running = False
            return

        # Open video capture in this thread
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
            last_odd_timestamp = -1

            print("Video processing loop started")

            while self.is_running and not self.force_stop_event.is_set():
                try:
                    # Update activity timestamp
                    self.last_activity_time = time.time()

                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video reached or failed to read frame")
                        # Emit video ended event to websocket
                        if self.socketio:
                            self.socketio.emit(
                                "video_ended",
                                {"reason": "end_of_video"},
                                namespace="/ws",
                            )
                        break

                    current_timestamp = frame_idx / fps if fps > 0 else 0

                    # Use exact same logic as standalone file
                    shifted_timestamp = current_timestamp - 0.5
                    inference_window_id = (
                        int(shifted_timestamp) if shifted_timestamp >= 0 else -1
                    )
                    is_in_inference_window = (
                        shifted_timestamp >= 0 and inference_window_id % 2 == 1
                    )

                    should_start_inference = (
                        is_in_inference_window
                        and inference_window_id != last_odd_timestamp
                        and not inference_active
                    )

                    if should_start_inference:
                        inference_active = True
                        inference_frame_count = 0
                        last_odd_timestamp = inference_window_id
                        self.tray_counter += 1

                    if inference_active:
                        inference_frame_count += 1
                        # Stop inference if we've completed minimum frames and moved out of inference window
                        if (
                            inference_frame_count >= self.config["min_inference_frames"]
                            and not is_in_inference_window
                        ):
                            inference_active = False

                    if inference_active:
                        vis, detection_count = self._process_inference_frame(
                            frame, inference_frame_count, current_timestamp
                        )
                    else:
                        vis = frame.copy()
                        detection_count = 0

                        # Show next inference time instead of "INFERENCE OFF"
                        if current_timestamp >= 0.5:
                            current_window = (
                                int(shifted_timestamp) if shifted_timestamp >= 0 else -1
                            )
                            if current_window % 2 == 0:  # Not in inference window
                                next_window = current_window + 1
                                next_inference_time = next_window + 1.5
                            else:  # In inference window but not active
                                next_inference_time = current_window + 1.5

                            cv2.putText(
                                vis,
                                f"Next inference: {next_inference_time:.1f}s",
                                (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                            )

                    fps_counter.update()
                    self.current_fps = fps_counter.get_fps()
                    self.inference_active = inference_active

                    self._add_fps_overlay(vis)
                    self._emit_frame(
                        vis,
                        {
                            "timestamp": current_timestamp,
                            "inference_active": inference_active,
                            "fps": self.current_fps,
                            "frame_idx": frame_idx,
                        },
                    )

                    frame_idx += 1

                    if self.shutdown_event.is_set() or self.force_stop_event.is_set():
                        print("Shutdown event received")
                        break

                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    # Continue processing other frames even if one fails
                    frame_idx += 1
                    continue

        except Exception as e:
            print(f"Critical error in video processing loop: {e}")
            import traceback

            traceback.print_exc()

            # Emit error to websocket
            if self.socketio:
                self.socketio.emit(
                    "video_ended",
                    {"reason": "critical_error", "error": str(e)},
                    namespace="/ws",
                )

        finally:
            print("Cleaning up video processing thread...")
            # Simple cleanup - just release video capture
            if self.cap:
                self.cap.release()
                self.cap = None

            # Completely empty the CUDA context stack
            if self.seg and hasattr(self.seg, "ctx"):
                try:
                    print("Emptying CUDA context stack...")
                    # Pop all contexts from the stack until it's empty
                    context_count = 0
                    while True:
                        try:
                            self.seg.ctx.pop()
                            context_count += 1
                            print(f"Popped context #{context_count}")
                        except Exception as pop_error:
                            print(
                                f"No more contexts to pop (popped {context_count} total)"
                            )
                            break

                    # Also try to detach the context
                    try:
                        self.seg.ctx.detach()
                        print("CUDA context detached")
                    except Exception as detach_error:
                        print(f"Context detach not needed or failed: {detach_error}")

                    print("CUDA context stack cleaned successfully")
                except Exception as e:
                    print(f"Warning: Error cleaning CUDA context stack: {e}")

            # Don't call explicit cleanup - let TRTSegmentor go out of scope naturally
            print("Processing thread ending naturally (context stack emptied)")
            self.is_running = False

    def _process_inference_frame(self, frame, inference_frame_count, current_timestamp):
        lb, scale, left, top = letterbox(
            frame, size=self.config["canvas_size"], pad_val=114
        )
        boxes, labels, scores, _ = self.seg.infer_fast(lb, measure_gpu=False)

        # Class-specific score filtering
        keep = np.zeros(len(scores), dtype=bool)
        for i in range(len(scores)):
            label_val = int(labels[i])
            if label_val == 0 and self.config["score_class0"] is not None:
                threshold = self.config["score_class0"]
            elif label_val == 1 and self.config["score_class1"] is not None:
                threshold = self.config["score_class1"]
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

        # Find indices in original arrays for mask copying
        if len(boxes_nms) > 0:
            keep_idx = []
            original_indices = np.where(keep)[0]

            for nms_box, nms_score, nms_label in zip(boxes_nms, scores_nms, labels_nms):
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

        if keep_idx.size > 0:
            masks_fp, _ = self.seg.copy_masks(keep_idx.astype(np.int64))
        else:
            masks_fp = np.empty((0, *self.seg.mask_hw), dtype=self.seg.dtype_masks)

        boxes_img = unletterbox_boxes(
            boxes_nms, scale, left, top, frame.shape[1], frame.shape[0]
        )
        masks_u8, _ = self._to_u8_fast(masks_fp)

        should_publish = inference_frame_count == 10

        if masks_u8.size > 0 and boxes_img.size > 0:
            vis, measurement_data = self._visualize_and_measure(
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
            detection_count = len(boxes_nms)
        else:
            vis = frame.copy()
            measurement_data = self._create_empty_measurement_data()
            detection_count = 0

        if should_publish and self.mqtt_client:
            self._publish_mqtt_data(measurement_data, current_timestamp)

        # Show detection count and frame info instead of "INFERENCE ON"
        cv2.putText(
            vis,
            f"Detections: {detection_count} | Frame: {inference_frame_count}/{self.config['min_inference_frames']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Add timestamp info
        cv2.putText(
            vis,
            f"Time: {current_timestamp:.1f}s | Tray: {self.tray_counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return vis, detection_count

    def _filter_by_score(self, scores, labels):
        keep = np.zeros(len(scores), dtype=bool)
        for i in range(len(scores)):
            label_val = int(labels[i])
            if label_val == 0 and self.config["score_class0"] is not None:
                threshold = self.config["score_class0"]
            elif label_val == 1 and self.config["score_class1"] is not None:
                threshold = self.config["score_class1"]
            else:
                threshold = self.config["score_threshold"]
            keep[i] = scores[i] >= threshold
        return keep

    def _get_keep_indices(
        self,
        keep,
        boxes_filtered,
        scores_filtered,
        labels_filtered,
        boxes_nms,
        scores_nms,
        labels_nms,
    ):
        if len(boxes_nms) == 0:
            return np.array([], dtype=np.int64)

        keep_idx = []
        original_indices = np.where(keep)[0]

        for nms_box, nms_score, nms_label in zip(boxes_nms, scores_nms, labels_nms):
            for j, orig_idx in enumerate(original_indices):
                if (
                    np.allclose(boxes_filtered[j], nms_box, atol=1e-5)
                    and np.isclose(scores_filtered[j], nms_score, atol=1e-5)
                    and labels_filtered[j] == nms_label
                ):
                    keep_idx.append(orig_idx)
                    break

        return np.array(keep_idx, dtype=np.int64)

    def _visualize_and_measure(
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
        vis = base_bgr.copy()
        H_img, W_img = vis.shape[:2]

        tray_start = 26
        tray_end = 1235

        objects = []
        for i, (box, lab) in enumerate(zip(boxes_img, labels)):
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
                    "left_edge": x1,
                    "right_edge": x2,
                    "width_px": width_px,
                    "length_px": length_px,
                }
            )

        objects.sort(key=lambda obj: obj["center_x"])

        class_0_count = sum(1 for obj in objects if int(obj["label"]) == 0)
        class_1_count = sum(1 for obj in objects if int(obj["label"]) == 1)
        missing_count = 0

        # Count missing baguettes based on gaps > 32px
        if len(objects) > 0:
            for i in range(len(objects) - 1):
                obj1 = objects[i]
                obj2 = objects[i + 1]
                distance = obj2["left_edge"] - obj1["right_edge"]
                if distance > 32.0:
                    missing_count += 1

        # Calculate average y-coordinate for distance lines
        avg_y = (
            int(np.mean([obj["center_y"] for obj in objects]))
            if objects
            else H_img // 2
        )

        # Draw masks and measurement lines for each object
        for obj_num, obj in enumerate(objects, 1):
            self._draw_object_visualization(
                vis, obj, masks_lb_u8, scale, left, top, W_img, H_img
            )

        # Draw distance measurement lines (white lines like standalone file)
        if len(objects) > 0:
            line_y = max(10, avg_y - 50)  # Draw line above the objects

            # Distance from tray start to first object
            first_obj = objects[0]
            line_start_x = max(0, min(tray_start, W_img - 1))
            line_end_x = max(0, min(int(first_obj["left_edge"]), W_img - 1))
            cv2.line(
                vis, (line_start_x, line_y), (line_end_x, line_y), (255, 255, 255), 1
            )

            # Distances between consecutive objects with missing detection
            for i in range(len(objects) - 1):
                obj1 = objects[i]
                obj2 = objects[i + 1]

                distance = obj2["left_edge"] - obj1["right_edge"]

                # Draw white line between object edges
                line_start_x = max(0, min(int(obj1["right_edge"]), W_img - 1))
                line_end_x = max(0, min(int(obj2["left_edge"]), W_img - 1))
                cv2.line(
                    vis,
                    (line_start_x, line_y),
                    (line_end_x, line_y),
                    (255, 255, 255),
                    1,
                )

                # Draw red cross (X) for missing baguettes
                if distance > 32.0:
                    center_x = int((line_start_x + line_end_x) / 2)
                    center_y = line_y

                    # Draw bright red cross (X shape)
                    cv2.line(
                        vis,
                        (center_x - 8, center_y - 8),
                        (center_x + 8, center_y + 8),
                        (0, 0, 255),
                        3,
                    )
                    cv2.line(
                        vis,
                        (center_x + 8, center_y - 8),
                        (center_x - 8, center_y + 8),
                        (0, 0, 255),
                        3,
                    )

            # Distance from last object to tray end
            last_obj = objects[-1]
            line_start_x = max(0, min(int(last_obj["right_edge"]), W_img - 1))
            line_end_x = max(0, min(tray_end, W_img - 1))
            cv2.line(
                vis, (line_start_x, line_y), (line_end_x, line_y), (255, 255, 255), 1
            )

        measurement_data = {
            "tray_number": self.tray_counter,
            "number_of_defected": class_0_count,
            "number_of_good": class_1_count,
            "number_of_missing": missing_count,
            "total_detected": len(objects),
            "timestamp": current_timestamp,
        }

        return vis, measurement_data

    def _draw_object_visualization(
        self, vis, obj, masks_lb_u8, scale, left, top, W_img, H_img
    ):
        i = obj["index"]
        box = obj["bbox"]
        lab = obj["label"]
        center_x, center_y = obj["center_x"], obj["center_y"]

        if i >= len(masks_lb_u8):
            return  # Safety check

        m = masks_lb_u8[i]
        x_lb, y_lb, w_lb, h_lb = cv2.boundingRect(m)
        if w_lb <= 0 or h_lb <= 0:
            return

        # Map letterbox coordinates to original image coordinates
        x1 = int(np.clip(round((x_lb - left) / scale), 0, W_img - 1))
        y1 = int(np.clip(round((y_lb - top) / scale), 0, H_img - 1))
        x2 = int(np.clip(round(((x_lb + w_lb) - left) / scale), 0, W_img))
        y2 = int(np.clip(round(((y_lb + h_lb) - top) / scale), 0, H_img))
        pw, ph = max(0, x2 - x1), max(0, y2 - y1)

        if pw == 0 or ph == 0:
            return

        # Resize mask patch to match image coordinates
        patch = m[y_lb : y_lb + h_lb, x_lb : x_lb + w_lb]

        # Ensure patch is not empty
        if patch.size == 0:
            return

        mroi = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_NEAREST)

        # Get the region of interest from the visualization
        roi = vis[y1 : y1 + ph, x1 : x1 + pw]

        # Ensure ROI dimensions match mask dimensions
        if roi.shape[:2] != mroi.shape:
            return

        # Color based on class: 0=Green visual, 1=Red visual (same as standalone file)
        label_val = int(lab)
        if label_val == 0:
            mask_color = (0, 0, 255)
            # Green in BGR for defected (class 0)
        else:
            mask_color = (0, 255, 0)  # Red in BGR for good (class 1)

        # Create colored overlay
        color_roi = np.full_like(roi, mask_color)

        # Apply alpha blending where mask is non-zero
        mask_bool = mroi > 0
        if np.any(mask_bool):
            # Create 3-channel mask for blending
            mask_3ch = np.stack([mask_bool, mask_bool, mask_bool], axis=2)

            # Blend only where mask is active
            blended = roi.copy()
            blended[mask_3ch] = (
                roi[mask_3ch] * (1.0 - self.config["alpha"])
                + color_roi[mask_3ch] * self.config["alpha"]
            ).astype(np.uint8)

            # Copy blended result back to vis
            roi[:] = blended

        # Draw measurement lines (same as before)
        x1b, y1b, x2b, y2b = [int(v) for v in box]
        center_x_int, center_y_int = int(center_x), int(center_y)

        # Blue horizontal line (length measurement)
        cv2.line(vis, (x1b, center_y_int), (x2b, center_y_int), (255, 0, 0), 1)
        # Black vertical line (width measurement)
        cv2.line(vis, (center_x_int, y1b), (center_x_int, y2b), (0, 0, 0), 1)

    def _create_empty_measurement_data(self):
        return {
            "tray_number": self.tray_counter,
            "number_of_defected": 0,
            "number_of_good": 0,
            "number_of_missing": 0,
            "total_detected": 0,
        }

    def _publish_mqtt_data(self, data, timestamp):
        payload = json.dumps(
            {**data, "timestamp": timestamp, "processing_time": time.time()}
        )

        if self.mqtt_client:
            self.mqtt_client.publish(self.config["mqtt_topic"], payload)

    def _add_fps_overlay(self, vis):
        height, width = vis.shape[:2]
        fps_text = f"FPS: {self.current_fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        # Move FPS counter down a bit - changed from (10, text_size[1] + 25) to (40, text_size[1] + 55)
        cv2.rectangle(
            vis,
            (width - text_size[0] - 20, 40),
            (width - 5, text_size[1] + 55),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            vis,
            fps_text,
            (width - text_size[0] - 15, text_size[1] + 50),
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


def apply_nms(boxes, scores, labels, nms_threshold=0.5):
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


class TRTSegmentor:
    def __init__(self, engine_path, verbose=False):
        print(f"Initializing TRTSegmentor with engine: {engine_path}")

        # Initialize CUDA - this creates a primary context for the current thread
        cuda.init()
        self.device = cuda.Device(0)
        # Make this the primary context for this device
        self.ctx = self.device.make_context()
        self.ctx.push()  # Make it current

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
            # Keep the context current for this thread - don't pop it here
            # The context will stay active for the lifetime of this object
            pass

    def infer_fast(self, lb_img_uint8, measure_gpu=False):
        # Context is already current for this thread, no need to push/pop
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
                self.host_buf[self.name_dets],
                self.dev_ptr[self.name_dets],
                self.stream,
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
        # Context is already current for this thread
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
        """
        Clean up all CUDA resources with timeout protection
        """
        print("Cleaning up TRTSegmentor CUDA resources...")

        # Set a timeout for the entire cleanup process
        cleanup_start_time = time.time()
        cleanup_timeout = 5.0  # 5 second timeout for cleanup

        try:
            # First, try to synchronize any pending operations with timeout
            try:
                if hasattr(self, "stream") and self.stream:
                    # Don't hang on synchronization
                    print("Synchronizing CUDA stream...")
                    self.stream.synchronize()
                    print("CUDA stream synchronized")
            except Exception as e:
                print(f"Warning: Could not synchronize CUDA stream: {e}")

            # Check timeout
            if time.time() - cleanup_start_time > cleanup_timeout:
                print("Cleanup timeout reached during stream sync - aborting")
                return

            # Free device memory with individual error handling
            print("Freeing device memory...")
            freed_count = 0
            for name, ptr in list(self.dev_ptr.items()):
                try:
                    if ptr:
                        ptr.free()
                        freed_count += 1
                except Exception as e:
                    print(f"Warning: Error freeing device memory for {name}: {e}")

                # Check timeout during cleanup
                if time.time() - cleanup_start_time > cleanup_timeout:
                    print(
                        f"Cleanup timeout reached - freed {freed_count}/{len(self.dev_ptr)} buffers"
                    )
                    break

            print(f"Freed {freed_count} device memory buffers")

            # Clear dictionaries
            self.host_buf.clear()
            self.dev_ptr.clear()
            self.pinned_flag.clear()

            # Reset other attributes
            self.mask_host_buf = None
            self.mask_host_capacity = 0

            print("CUDA resources cleanup attempted")

        except Exception as e:
            print(f"Warning: Error during CUDA resource cleanup: {e}")

        finally:
            # Always try to detach context, even if other cleanup failed
            print("Detaching CUDA context...")
            try:
                if hasattr(self, "ctx") and self.ctx:
                    # Don't let context cleanup hang the system
                    self.ctx.pop()
                    self.ctx.detach()
                    print("CUDA context detached successfully")
            except Exception as e:
                # This is the problematic TensorRT error - don't let it crash the app
                print(f"Warning: TensorRT context cleanup error (IGNORED): {e}")
                print("This TensorRT cleanup warning can be safely ignored")

            # Small delay to let any background operations complete
            try:
                time.sleep(0.1)
            except:
                pass

            print("TensorRT cleanup completed (with warnings ignored)")
