#!/usr/bin/env python3

import argparse
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from collections import deque, defaultdict
import json
import os
import threading
import queue
import base64
import csv
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

PINNED_THRESHOLD_BYTES = 16 * 1024 * 1024
TARGET_FPS = 30.0

COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

def clamp_roi(cx, cy, w, h, img_w, img_h):
    x1 = int(round(cx - w / 2)); y1 = int(round(cy - h / 2))
    x2 = x1 + int(w);            y2 = y1 + int(h)
    x1 = max(0, min(img_w - 1, x1)); y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2));     y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI after clamping.")
    return x1, y1, x2, y2

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
    out = boxes_xyxy.astype(np.float32, copy=False)
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - left) / scale, 0, out_w - 1)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top) / scale, 0, out_h - 1)
    return out

def unletterbox_masks_roi(masks_u8, boxes_img, scale, left, top, out_w, out_h, max_workers=4):
    if masks_u8.size == 0:
        return masks_u8, 0.0, 0.0

    t0 = time.perf_counter()
    N, Hm, Wm = masks_u8.shape
    out_masks = np.zeros((N, out_h, out_w), dtype=np.uint8)

    boxes_img = boxes_img.astype(np.float32, copy=False)

    x1 = np.clip(left + boxes_img[:, 0] * scale, 0, Wm).astype(np.int32)
    y1 = np.clip(top  + boxes_img[:, 1] * scale, 0, Hm).astype(np.int32)
    x2 = np.clip(left + boxes_img[:, 2] * scale, 0, Wm).astype(np.int32)
    y2 = np.clip(top  + boxes_img[:, 3] * scale, 0, Hm).astype(np.int32)

    dx1 = np.clip(boxes_img[:, 0], 0, out_w).astype(np.int32)
    dy1 = np.clip(boxes_img[:, 1], 0, out_h).astype(np.int32)
    dx2 = np.clip(boxes_img[:, 2], 0, out_w).astype(np.int32)
    dy2 = np.clip(boxes_img[:, 3], 0, out_h).astype(np.int32)

    t1 = time.perf_counter()
    prep_time = t1 - t0

    def place(i):
        if x2[i] <= x1[i] or y2[i] <= y1[i] or dx2[i] <= dx1[i] or dy2[i] <= dy1[i]:
            return
        src = masks_u8[i, y1[i]:y2[i], x1[i]:x2[i]]
        if src.size == 0:
            return
        w_dst = int(dx2[i] - dx1[i])
        h_dst = int(dy2[i] - dy1[i])
        dst_roi = cv2.resize(src, (w_dst, h_dst), interpolation=cv2.INTER_NEAREST)
        out_masks[i, dy1[i]:dy2[i], dx1[i]:dx2[i]] = dst_roi

    t2 = time.perf_counter()
    if N > 3:
        with ThreadPoolExecutor(max_workers=min(max_workers, N)) as ex:
            list(ex.map(place, range(N)))
    else:
        for i in range(N):
            place(i)
    t3 = time.perf_counter()
    resize_time = t3 - t2

    return out_masks, prep_time, resize_time

def tray_is_complete(mask_u8, margin_px=2):
    if mask_u8.size == 0:
        return False
    h, w = mask_u8.shape
    if (mask_u8[:margin_px, :].any() or mask_u8[-margin_px:, :].any() or
        mask_u8[:, :margin_px].any() or mask_u8[:, -margin_px:].any()):
        return False
    return mask_u8.any()

def pick_complete_tray(tray_masks):
    best_idx, best_area = -1, 0
    for i, m in enumerate(tray_masks):
        if tray_is_complete(m, margin_px=2):
            area = int((m > 0).sum())
            if area > best_area:
                best_idx, best_area = i, area
    return best_idx

def mask_iou(a_u8, b_u8):
    if a_u8 is None or b_u8 is None:
        return 0.0
    if a_u8.shape != b_u8.shape:
        return 0.0
    a = (a_u8 > 0).astype(np.uint8)
    b = (b_u8 > 0).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def order_baguettes_row_major(bag_items):
    def key_fn(item):
        x1,y1,x2,y2 = item["bbox"]
        cx = (x1+x2)*0.5
        cy = (y1+y2)*0.5
        return (cy, cx)
    return sorted(bag_items, key=key_fn)

def visualize_baguettes_and_tray(full_frame_bgr, roi_rect, baguette_masks_roi, tray_mask_roi, alpha=0.45):
    rx1, ry1, rx2, ry2 = roi_rect
    vis = full_frame_bgr

    if tray_mask_roi is not None and tray_mask_roi.any():
        contours, _ = cv2.findContours(tray_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt + np.array([[rx1, ry1]], dtype=np.int32)
            cv2.polylines(vis, [cnt], isClosed=True, color=(255, 255, 255), thickness=2)

    if len(baguette_masks_roi) > 0:
        roi_view = vis[ry1:ry2, rx1:rx2]
        base = roi_view.copy()
        green = np.zeros_like(roi_view)
        green[:, :, 1] = 255
        union = np.zeros((roi_view.shape[0], roi_view.shape[1]), dtype=np.uint8)
        for m in baguette_masks_roi:
            if m.shape[:2] != union.shape:
                m = cv2.resize(m, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_NEAREST)
            union = cv2.bitwise_or(union, (m > 0).astype(np.uint8))
        blended = cv2.addWeighted(base, 1.0 - alpha, green, alpha, 0.0)
        cv2.copyTo(blended, union, roi_view)

    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 0, 0), 2)

    return vis

def draw_measurement_lines(frame_bgr, roi_rect, bag_items, draw_height, draw_width):
    rx1, ry1, rx2, ry2 = roi_rect
    for item in bag_items:
        x1,y1,x2,y2 = item["bbox"]
        fx1, fy1, fx2, fy2 = int(rx1 + x1), int(ry1 + y1), int(rx1 + x2), int(ry1 + y2)
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        if draw_height:
            cv2.line(frame_bgr, (cx, fy1), (cx, fy2), (255,255,255), 1)
        if draw_width:
            cv2.line(frame_bgr, (fx1, cy), (fx2, cy), (255,255,255), 1)
    return frame_bgr

def load_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    roi = cfg.get("ROI", {})
    cx = int(float(roi.get("x", "0")))
    cy = int(float(roi.get("y", "0")))
    h  = int(float(roi.get("height", "0")))
    w  = int(float(roi.get("width", "0")))
    ref = cfg.get("reference", {})
    pix_len = float(ref.get("pixel_length", "1"))
    act_len = float(ref.get("actual_length", "1"))
    mm_per_px = act_len / pix_len if pix_len != 0 else 0.0
    unit = ref.get("unit", "mm")
    return (cx, cy, w, h), mm_per_px, unit

def parse_button_array(s):
    if s is None:
        return []
    txt = s.strip()
    if not txt:
        return []
    try:
        inner = txt.strip()
        if inner.startswith('[') and inner.endswith(']'):
            inner_content = inner[1:-1].strip()
            if inner_content and all(ch not in inner_content for ch in "'\""):
                parts = [p.strip() for p in inner_content.split(',') if p.strip()]
                quoted = "[" + ",".join(f"\"{p}\"" for p in parts) + "]"
                arr = json.loads(quoted)
            else:
                arr = json.loads(inner.replace("'", "\""))
        else:
            arr = json.loads(txt.replace("'", "\""))
        res = [str(x).lower() for x in arr if isinstance(x, (str, int, float))]
    except Exception:
        if txt.startswith('[') and txt.endswith(']'):
            txt = txt[1:-1]
        res = [p.strip().lower().strip('\'"') for p in txt.split(',') if p.strip()]
    allowed = {'height', 'width'}
    return [x for x in res if x in allowed]

def fmt_measurement(px_val: int, scale_units_per_px, unit_label: str):
    if scale_units_per_px is None:
        return f"{int(px_val)}px"
    else:
        return f"{int(px_val)}px ({int(round(px_val*scale_units_per_px))}{unit_label})"

class OptimizedTRTSegmentor:
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

            self._setup_io()
            self.stream = cuda.Stream()
            self.start_evt = cuda.Event()
            self.end_evt = cuda.Event()

            self.output_boxes = np.empty((self.max_det, 4), dtype=np.float32)
            self.output_labels = np.empty(self.max_det, dtype=np.int32)
            self.output_scores = np.empty(self.max_det, dtype=np.float32)

            self.mask_host_capacity = 0
            self.mask_host_buf = None

            self._thresh_mod = SourceModule(r"""
            extern "C" __global__
            void thresh_u8(const float* __restrict__ src,
                           unsigned char* __restrict__ dst,
                           int n, int src_offset, int dst_offset, float thr) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i < n) {
                    float v = src[src_offset + i];
                    dst[dst_offset + i] = (unsigned char)((v > thr) ? 255 : 0);
                }
            }
            """)
            self._k_thresh = self._thresh_mod.get_function("thresh_u8")

            self.dev_masks_u8 = None
            self.u8_capacity = 0
            self.mask_host_buf_u8 = None

            self._roi_mod = SourceModule(r"""
            extern "C" __global__
            void resize_thresh_roi(
                const float* __restrict__ src,
                int Hm, int Wm,
                int src_offset_elems,
                int sx, int sy, int sw, int sh,
                unsigned char* __restrict__ dst,
                int dst_offset_elems,
                int dw, int dh,
                float thr
            ){
                int x = blockDim.x * blockIdx.x + threadIdx.x;
                int y = blockDim.y * blockIdx.y + threadIdx.y;
                if (x >= dw || y >= dh) return;

                float fx = ((x + 0.5f) * (float)sw / (float)dw) - 0.5f;
                float fy = ((y + 0.5f) * (float)sh / (float)dh) - 0.5f;
                int ix = sx + (int)roundf(fx);
                int iy = sy + (int)roundf(fy);
                ix = max(0, min(Wm - 1, ix));
                iy = max(0, min(Hm - 1, iy));

                float v = src[src_offset_elems + iy * Wm + ix];
                dst[dst_offset_elems + y * dw + x] = (unsigned char)(v > thr ? 255 : 0);
            }
            """)
            self._k_roi = self._roi_mod.get_function("resize_thresh_roi")

            self.dev_rois_u8 = None
            self.host_rois_u8 = None
            self.roi_bytes_capacity = 0

            self._warmup_cv2()
            
            print("TRTSegmentor initialization complete")
            
        finally:
            pass

    def _setup_io(self):
        if self.trt10:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        def find_tensor(keywords):
            for name in names:
                if any(kw in name.lower() for kw in keywords):
                    return name
            return None
        self.name_in = find_tensor(['raw_input', 'input'])
        self.name_dets = find_tensor(['det', 'boxes'])
        self.name_labels = find_tensor(['label', 'class'])
        self.name_masks = find_tensor(['mask', 'seg'])
        if any(x is None for x in [self.name_in, self.name_dets, self.name_labels, self.name_masks]):
            raise RuntimeError("Could not find required tensors")
        if self.trt10:
            self.dtype_in = trt.nptype(self.engine.get_tensor_dtype(self.name_in))
            self.dtype_dets = trt.nptype(self.engine.get_tensor_dtype(self.name_dets))
            self.dtype_labels = trt.nptype(self.engine.get_tensor_dtype(self.name_labels))
            self.dtype_masks = trt.nptype(self.engine.get_tensor_dtype(self.name_masks))
        else:
            def get_dtype(name):
                idx = self.engine.get_binding_index(name)
                return trt.nptype(self.engine.get_binding_dtype(idx))
            self.dtype_in = get_dtype(self.name_in)
            self.dtype_dets = get_dtype(self.name_dets)
            self.dtype_labels = get_dtype(self.name_labels)
            self.dtype_masks = get_dtype(self.name_masks)
        self._allocate_buffers()

    def _allocate_buffers(self):
        self.input_shape = (1, 640, 640, 3)
        self.max_det = 100
        self.mask_hw = (640, 640)
        self.dev_ptr, self.host_buf, self.pinned_flag = {}, {}, {}

        def allocate(name, shape, dtype, force_pinned=False):
            numel = int(np.prod(shape))
            nbytes = numel * np.dtype(dtype).itemsize
            use_pinned = force_pinned or (nbytes >= PINNED_THRESHOLD_BYTES)
            if use_pinned:
                buf = cuda.pagelocked_empty(numel, dtype)
            else:
                buf = np.empty(numel, dtype=dtype)
            self.host_buf[name] = buf
            self.pinned_flag[name] = use_pinned
            self.dev_ptr[name] = cuda.mem_alloc(nbytes)

        allocate(self.name_in, self.input_shape, self.dtype_in)
        allocate(self.name_dets, (1, self.max_det, 5), self.dtype_dets, force_pinned=True)
        allocate(self.name_labels, (1, self.max_det), self.dtype_labels, force_pinned=True)

        mask_shape = (1, self.max_det, *self.mask_hw)
        mask_bytes = int(np.prod(mask_shape)) * np.dtype(self.dtype_masks).itemsize
        self.dev_ptr[self.name_masks] = cuda.mem_alloc(mask_bytes)

        if self.trt10:
            for name, ptr in self.dev_ptr.items():
                self.context.set_tensor_address(name, int(ptr))
        else:
            self.bindings = [None] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.bindings[i] = int(self.dev_ptr[name])

    def _warmup_cv2(self):
        dummy = np.zeros((100, 100), dtype=np.uint8)
        _ = cv2.resize(dummy, (200, 200), interpolation=cv2.INTER_NEAREST)

    def _ensure_u8_capacity(self, needed_masks):
        Hm, Wm = self.mask_hw
        max_reasonable_capacity = 200
        needed_capacity = min(needed_masks, max_reasonable_capacity)
        
        if needed_capacity <= getattr(self, "u8_capacity", 0): return
        
        current_cap = getattr(self, "u8_capacity", 0)
        if current_cap > needed_capacity * 4:
            needed_capacity = max(needed_capacity * 2, 32)
        else:
            needed_capacity = max(needed_capacity * 2, 32)
            
        needed_capacity = min(needed_capacity, max_reasonable_capacity)
        
        self.u8_capacity = needed_capacity
        bytes_u8 = self.u8_capacity * Hm * Wm
        if getattr(self, "dev_masks_u8", None) is not None: self.dev_masks_u8.free()
        self.dev_masks_u8 = cuda.mem_alloc(bytes_u8)
        self.mask_host_buf_u8 = cuda.pagelocked_empty((self.u8_capacity, Hm, Wm), dtype=np.uint8)

    def _ensure_roi_bytes(self, needed_bytes):
        max_reasonable_bytes = 50 * 1024 * 1024
        needed_bytes = min(needed_bytes, max_reasonable_bytes)
        
        if needed_bytes <= getattr(self, "roi_bytes_capacity", 0): return
        
        current_cap = getattr(self, "roi_bytes_capacity", 0)
        if current_cap > needed_bytes * 4:
            needed_bytes = max(needed_bytes * 2, 1 << 20)
        else:
            needed_bytes = max(needed_bytes * 2, 1 << 20)
            
        needed_bytes = min(needed_bytes, max_reasonable_bytes)
        
        self.roi_bytes_capacity = needed_bytes
        if getattr(self, "dev_rois_u8", None) is not None: self.dev_rois_u8.free()
        self.dev_rois_u8 = cuda.mem_alloc(self.roi_bytes_capacity)
        self.host_rois_u8 = cuda.pagelocked_empty(self.roi_bytes_capacity, dtype=np.uint8)

    def infer_optimized(self, img_uint8):
        np.copyto(self.host_buf[self.name_in], img_uint8.ravel())
        cuda.memcpy_htod_async(self.dev_ptr[self.name_in], self.host_buf[self.name_in], self.stream)
        self.start_evt.record(self.stream)
        if self.trt10:
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.end_evt.record(self.stream)
        cuda.memcpy_dtoh_async(self.host_buf[self.name_dets], self.dev_ptr[self.name_dets], self.stream)
        cuda.memcpy_dtoh_async(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels], self.stream)
        self.stream.synchronize()

        dets_raw = self.host_buf[self.name_dets].reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].reshape(1, self.max_det)[0]
        self.output_boxes[...] = dets_raw[:, :4]
        self.output_scores[...] = dets_raw[:, 4]
        self.output_labels[...] = labels_raw.astype(np.int32, copy=False)
        return self.output_boxes, self.output_labels, self.output_scores, {}

    def copy_masks_optimized(self, indices, thr=0.5):
        timings = {"d2h_s": 0.0, "gpu_thresh_s": 0.0, "total_s": 0.0}
        if len(indices) == 0:
            return np.empty((0, *self.mask_hw), dtype=self.dtype_masks), timings

        t_total0 = time.perf_counter()
        Hm, Wm = self.mask_hw
        n_masks = len(indices)
        n_pix = Hm * Wm
        itemsize = np.dtype(self.dtype_masks).itemsize
        slice_bytes = Hm * Wm * itemsize
        base_addr = int(self.dev_ptr[self.name_masks])

        if self.dtype_masks == np.uint8:
            if self.mask_host_buf is None or self.mask_host_capacity < n_masks:
                self.mask_host_capacity = max(n_masks * 2, 32)
                self.mask_host_buf = cuda.pagelocked_empty((self.mask_host_capacity, Hm, Wm), dtype=self.dtype_masks)
            t0 = time.perf_counter()
            if n_masks > 1 and np.all(np.diff(indices) == 1):
                src = int(base_addr + int(indices[0]) * slice_bytes)
                cuda.memcpy_dtoh_async(self.mask_host_buf[:n_masks].ravel(), src, stream=self.stream)
            else:
                for k, idx in enumerate(indices):
                    src = int(base_addr + int(idx) * slice_bytes)
                    dst = self.mask_host_buf[k].ravel()
                    cuda.memcpy_dtoh_async(dst, src, stream=self.stream)
            self.stream.synchronize()
            timings["d2h_s"] = time.perf_counter() - t0
            timings["total_s"] = time.perf_counter() - t_total0
            return self.mask_host_buf[:n_masks].copy(), timings

        self._ensure_u8_capacity(n_masks)
        threads = 256
        t_gpu0 = time.perf_counter()
        for k, idx in enumerate(indices):
            src_offset = int(idx) * (Hm * Wm)
            dst_offset = k * (Hm * Wm)
            grid = (((Hm * Wm) + threads - 1) // threads, 1, 1)
            self._k_thresh(
                self.dev_ptr[self.name_masks],
                self.dev_masks_u8,
                np.int32(Hm * Wm),
                np.int32(src_offset),
                np.int32(dst_offset),
                np.float32(thr),
                block=(threads, 1, 1), grid=grid, stream=self.stream
            )
        self.stream.synchronize()
        timings["gpu_thresh_s"] = time.perf_counter() - t_gpu0

        t_d2h0 = time.perf_counter()
        cuda.memcpy_dtoh_async(self.mask_host_buf_u8[:n_masks].ravel(), int(self.dev_masks_u8), stream=self.stream)
        self.stream.synchronize()
        timings["d2h_s"] = time.perf_counter() - t_d2h0
        timings["total_s"] = time.perf_counter() - t_total0
        return self.mask_host_buf_u8[:n_masks].copy(), timings

    def copy_and_resize_masks_roi_gpu(self, indices, boxes_img, scale, left, top, out_w, out_h, thr=0.5):
        timings = {"prep_cpu_s": 0.0, "kernel_s": 0.0, "d2h_s": 0.0, "paste_cpu_s": 0.0, "total_s": 0.0}
        if len(indices) == 0:
            return np.empty((0, out_h, out_w), dtype=np.uint8), timings
        if self.dtype_masks != np.float32:
            raise RuntimeError("GPU ROI path expects float32 mask output from engine.")

        t_total0 = time.perf_counter()
        Hm, Wm = self.mask_hw
        n_pix = Hm * Wm
        N = len(indices)
        boxes_img = boxes_img.astype(np.float32, copy=False)

        t_p0 = time.perf_counter()
        sx = np.clip(left + boxes_img[:,0] * scale, 0, Wm).astype(np.int32)
        sy = np.clip(top  + boxes_img[:,1] * scale, 0, Hm).astype(np.int32)
        ex = np.clip(left + boxes_img[:,2] * scale, 0, Wm).astype(np.int32)
        ey = np.clip(top  + boxes_img[:,3] * scale, 0, Hm).astype(np.int32)
        sw = np.maximum(ex - sx, 0).astype(np.int32)
        sh = np.maximum(ey - sy, 0).astype(np.int32)

        dx = np.clip(boxes_img[:,0], 0, out_w).astype(np.int32)
        dy = np.clip(boxes_img[:,1], 0, out_h).astype(np.int32)
        ex2 = np.clip(boxes_img[:,2], 0, out_w).astype(np.int32)
        ey2 = np.clip(boxes_img[:,3], 0, out_h).astype(np.int32)
        dw = np.maximum(ex2 - dx, 0).astype(np.int32)
        dh = np.maximum(ey2 - dy, 0).astype(np.int32)

        sizes = (dw * dh).astype(np.int64)
        valid = (sw > 0) & (sh > 0) & (dw > 0) & (dh > 0)
        sizes[~valid] = 0
        offsets = np.zeros(N, dtype=np.int64)
        if N > 0:
            np.cumsum(sizes[:-1], out=offsets[1:])
        total_elems = int(offsets[-1] + sizes[-1]) if N > 0 else 0
        timings["prep_cpu_s"] = time.perf_counter() - t_p0

        self._ensure_roi_bytes(total_elems if total_elems > 0 else 1)

        t_k0 = time.perf_counter()
        masks_full = np.zeros((N, out_h, out_w), dtype=np.uint8)
        block = (16, 16, 1)
        for i, idx in enumerate(indices):
            if sizes[i] == 0:
                continue
            src_offset = int(idx) * n_pix
            dst_offset = int(offsets[i])
            grid = ( (int(dw[i]) + block[0]-1)//block[0],
                     (int(dh[i]) + block[1]-1)//block[1], 1 )
            self._k_roi(
                self.dev_ptr[self.name_masks],
                np.int32(Hm), np.int32(Wm),
                np.int32(src_offset),
                np.int32(int(sx[i])), np.int32(int(sy[i])),
                np.int32(int(sw[i])), np.int32(int(sh[i])),
                self.dev_rois_u8,
                np.int32(dst_offset),
                np.int32(int(dw[i])), np.int32(int(dh[i])),
                np.float32(thr),
                block=block, grid=grid, stream=self.stream
            )
        self.stream.synchronize()
        timings["kernel_s"] = time.perf_counter() - t_k0

        t_d0 = time.perf_counter()
        if total_elems > 0:
            cuda.memcpy_dtoh_async(self.host_rois_u8[:total_elems], int(self.dev_rois_u8), stream=self.stream)
        self.stream.synchronize()
        timings["d2h_s"] = time.perf_counter() - t_d0

        t_paste0 = time.perf_counter()
        base = self.host_rois_u8
        for i in range(N):
            if sizes[i] == 0:
                continue
            count = int(sizes[i]); off = int(offsets[i])
            roi = np.frombuffer(base, dtype=np.uint8, count=count, offset=off).reshape(int(dh[i]), int(dw[i]))
            masks_full[i, dy[i]:dy[i]+dh[i], dx[i]:dx[i]+dw[i]] = roi
        timings["paste_cpu_s"] = time.perf_counter() - t_paste0

        timings["total_s"] = time.perf_counter() - t_total0
        return masks_full, timings

    def cleanup(self):
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

            print("Cleaning up TensorRT engine and CUDA context...")

            if hasattr(self, "context") and self.context:
                print("Destroying TensorRT execution context...")
                del self.context
                self.context = None

            if hasattr(self, "engine") and self.engine:
                print("Destroying TensorRT engine...")
                del self.engine
                self.engine = None

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

            if hasattr(self, "dev_masks_u8") and self.dev_masks_u8:
                self.dev_masks_u8.free()
            if hasattr(self, "dev_rois_u8") and self.dev_rois_u8:
                self.dev_rois_u8.free()

            print("CUDA resources cleanup attempted")

        except Exception as e:
            print(f"Warning: Error during CUDA resource cleanup: {e}")

        finally:
            print("Cleaning up CUDA context stack...")
            try:
                if hasattr(self, "ctx") and self.ctx:
                    context_count = 0
                    while True:
                        try:
                            self.ctx.pop()
                            context_count += 1
                            print(f"Popped context #{context_count}")
                        except Exception:
                            print(f"No more contexts to pop (popped {context_count} total)")
                            break

                    try:
                        self.ctx.detach()
                        print("CUDA context detached")
                    except Exception as detach_error:
                        print(f"Context detach not needed or failed: {detach_error}")

                    print("CUDA context stack cleaned successfully")
            except Exception as e:
                print(f"Warning: TensorRT context cleanup error (IGNORED): {e}")
                print("This TensorRT cleanup warning can be safely ignored")

            try:
                time.sleep(0.1)
            except:
                pass

            print("TensorRT cleanup completed (with warnings ignored)")

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
        
        self.prev_tray_mask_ref = None
        self.tray_stable_start_time = None
        self.current_tray_logged = False
        self.roi_config = None
        self.button_list = []

    def set_socketio(self, socketio):
        self.socketio = socketio

    def set_mqtt(self, mqtt_client):
        self.mqtt_client = mqtt_client

    def initialize_from_config(self, config):
        self.config = {
            "engine_path": config.get("engine_path", ""),
            "video_path": config.get("video_path", ""),
            "config_path": config.get("config_path", ""),
            "button": config.get("button", "[]"),
            "score_threshold": config.get("score_threshold", 0.7),
            "mask_threshold": config.get("mask_threshold", 0.7),
            "canvas_size": config.get("canvas_size", 640),
            "alpha": config.get("alpha", 0.45),
            "target_fps": config.get("target_fps", TARGET_FPS),
            "mqtt_topic": config.get("mqtt_topic", "baguette/results"),
            "csv_output_dir": config.get("csv_output_dir", None),
            "stability_seconds": config.get("stability_seconds", 0.4),
            "tray_iou_same": config.get("tray_iou_same", 0.6),
            "gpu_roi": config.get("gpu_roi", True),
            "engine_binary_masks": config.get("engine_binary_masks", True),
            "workers": config.get("workers", 4),
        }

        if not os.path.exists(self.config["engine_path"]):
            print(f"Engine path does not exist: {self.config['engine_path']}")
            return False

        if not os.path.exists(self.config["config_path"]):
            print(f"Config path does not exist: {self.config['config_path']}")
            return False

        try:
            roi_tuple, mm_per_px, unit = load_config(self.config["config_path"])
            self.roi_config = {
                "cx": roi_tuple[0], "cy": roi_tuple[1], "w": roi_tuple[2], "h": roi_tuple[3],
                "mm_per_px": mm_per_px, "unit": unit
            }
            self.button_list = parse_button_array(self.config["button"])
        except Exception as e:
            print(f"Error loading config: {e}")
            return False

        return True

    def _initialize_engine_in_thread(self):
        try:
            print("Initializing TensorRT engine in processing thread...")
            self.seg = OptimizedTRTSegmentor(self.config["engine_path"])
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

    def _add_csv_data(self, tray_number, baguette_number, height_measurement, width_measurement):
        csv_row = {
            'tray_number': tray_number,
            'baguette_number': baguette_number,
            'height': height_measurement,
            'width': width_measurement
        }
        self.csv_data.append(csv_row)
        print(f"Added to CSV: {csv_row}")

    def _save_csv_data(self):
        if not self.csv_data:
            print("No CSV data to save")
            return False

        try:
            output_dir = self.config.get("csv_output_dir", "output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            csv_filename = f"baguette_measurements_{timestamp_str}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)

            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['tray_number', 'baguette_number', 'height', 'width']
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
        return {"data": self.csv_data, "row_count": len(self.csv_data)}

    def emergency_shutdown(self):
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
        
        self.prev_tray_mask_ref = None
        self.tray_stable_start_time = None
        self.current_tray_logged = False

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
        self.reset_to_initial_state()

    def is_stuck(self):
        return (time.time() - self.last_activity_time) > self.processing_timeout

    def _process_video(self):
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

            do_height = 'height' in self.button_list
            do_width = 'width' in self.button_list

            fps_counter = FPSCounter()
            frame_idx = 0
            frame_interval = 1.0 / max(1e-6, self.config["target_fps"])

            print("Video processing loop started")

            while self.is_running and not self.force_stop_event.is_set():
                try:
                    frame_t0 = time.perf_counter()
                    self.last_activity_time = time.time()

                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video reached - initiating proper shutdown")
                        self._save_csv_data()
                        if self.socketio:
                            self.socketio.emit(
                                "video_ended",
                                {"reason": "end_of_video"},
                                namespace="/ws",
                            )
                        self.is_running = False
                        self.shutdown_event.set()
                        break

                    current_video_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    current_video_time_seconds = current_video_time_ms / 1000.0

                    vis = self._process_frame(
                        frame, current_video_time_seconds, width, height,
                        do_height, do_width, frame_idx, frame_count
                    )

                    fps_counter.update()
                    self.current_fps = fps_counter.get_fps()

                    self._emit_frame(
                        vis,
                        {
                            "timestamp": current_video_time_seconds,
                            "fps": self.current_fps,
                            "frame_idx": frame_idx,
                            "tray_number": self.tray_counter,
                        },
                    )

                    frame_idx += 1

                    sleep_t = frame_interval - (time.perf_counter() - frame_t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                    if frame_idx % 100 == 0:
                        import gc
                        gc.collect()

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
                    self.seg.cleanup()
                except Exception as e:
                    print(f"Warning: Error during segmentor cleanup: {e}")
                finally:
                    self.seg = None

            import gc
            gc.collect()
            print("Processing thread ending naturally (full cleanup completed)")
            self.is_running = False

    def _process_frame(self, frame, current_video_time_seconds, width, height, 
                      do_height, do_width, frame_idx, frame_count):
        try:
            rx1, ry1, rx2, ry2 = clamp_roi(self.roi_config["cx"], self.roi_config["cy"], 
                                          self.roi_config["w"], self.roi_config["h"], width, height)
            roi = frame[ry1:ry2, rx1:rx2].copy()
            roi_h_, roi_w_ = roi.shape[:2]

            lb, scale, left, top = letterbox(roi, size=self.config["canvas_size"], pad_val=114)

            boxes, labels, scores, _ = self.seg.infer_optimized(lb)

            keep = scores >= self.config["score_threshold"]
            boxes_kept = boxes[keep]
            labels_kept = labels[keep]
            keep_idx = np.flatnonzero(keep)

            boxes_roi = unletterbox_boxes(boxes_kept, scale, left, top, roi_w_, roi_h_)

            if keep_idx.size > 0:
                if self.config["gpu_roi"] and self.seg.dtype_masks == np.float32:
                    masks_roi, _ = self.seg.copy_and_resize_masks_roi_gpu(
                        keep_idx, boxes_roi, scale, left, top, roi_w_, roi_h_, thr=self.config["mask_threshold"]
                    )
                else:
                    masks_fp, _ = self.seg.copy_masks_optimized(keep_idx, thr=self.config["mask_threshold"])
                    if self.config["engine_binary_masks"]:
                        masks_u8 = masks_fp.astype(np.uint8, copy=False)
                    else:
                        masks_u8 = ((masks_fp > self.config["mask_threshold"]) * 255).astype(np.uint8) if masks_fp.size > 0 else masks_fp.astype(np.uint8)
                    masks_roi, _, _ = unletterbox_masks_roi(
                        masks_u8, boxes_roi, scale, left, top, roi_w_, roi_h_, max_workers=self.config["workers"]
                    )
                    del masks_fp, masks_u8
            else:
                masks_roi = np.empty((0, roi_h_, roi_w_), dtype=np.uint8)

            baguette_items = []
            tray_masks = []
            for m, lab, bx in zip(masks_roi, labels_kept, boxes_roi):
                if lab == 0:
                    baguette_items.append({"mask": m, "bbox": bx})
                elif lab == 1:
                    tray_masks.append(m)

            chosen_tray_idx = pick_complete_tray(tray_masks)
            tray_mask = tray_masks[chosen_tray_idx] if chosen_tray_idx != -1 else None

            filtered_baguettes = []
            if tray_mask is not None:
                tray_bin = (tray_mask > 0).astype(np.uint8)
                for item in baguette_items:
                    bm = item["mask"]
                    if bm is None or bm.size == 0:
                        continue
                    b_bin = (bm > 0).astype(np.uint8)
                    inter = cv2.bitwise_and(b_bin, tray_bin)
                    b_area = int(b_bin.sum())
                    if b_area > 0 and int(inter.sum()) >= 0.5 * b_area:
                        filtered_baguettes.append({"mask": (b_bin * 255).astype(np.uint8),
                                                   "bbox": item["bbox"]})

            now = time.perf_counter()
            measurement_draw_enabled = False
            if tray_mask is not None:
                if self.prev_tray_mask_ref is None:
                    self.prev_tray_mask_ref = tray_mask.copy()
                    self.tray_stable_start_time = now
                    self.current_tray_logged = False
                else:
                    iou = mask_iou(tray_mask, self.prev_tray_mask_ref)
                    if iou >= self.config["tray_iou_same"]:
                        if self.tray_stable_start_time is None:
                            self.tray_stable_start_time = now
                        duration = now - self.tray_stable_start_time
                        measurement_draw_enabled = True
                        if (not self.current_tray_logged) and duration >= self.config["stability_seconds"]:
                            self.tray_counter += 1
                            count = len(filtered_baguettes)
                            print(f"\nTray {self.tray_counter} baguettes: {count}")
                            if count > 0 and (do_height or do_width):
                                ordered = order_baguettes_row_major(filtered_baguettes)
                                for idx, itm in enumerate(ordered, start=1):
                                    x1,y1,x2,y2 = itm["bbox"]
                                    h_px = int(round((y2 - y1)))
                                    w_px = int(round((x2 - x1)))
                                    
                                    height_str = fmt_measurement(h_px, self.roi_config["mm_per_px"], self.roi_config["unit"]) if do_height else None
                                    width_str = fmt_measurement(w_px, self.roi_config["mm_per_px"], self.roi_config["unit"]) if do_width else None
                                    
                                    parts = []
                                    if height_str: parts.append(f"height={height_str}")
                                    if width_str: parts.append(f"width={width_str}")
                                    if parts:
                                        print(f"b{idx}: " + " ".join(parts))
                                    
                                    self._add_csv_data(self.tray_counter, idx, height_str, width_str)
                                    
                                    if self.mqtt_client:
                                        mqtt_payload = {
                                            'tray_number': self.tray_counter,
                                            'baguette_number': idx,
                                            'height': height_str,
                                            'width': width_str,
                                            'timestamp': time.time()
                                        }
                                        payload_json = json.dumps(mqtt_payload)
                                        self.mqtt_client.publish(self.config["mqtt_topic"], payload_json)
                            
                            self.current_tray_logged = True
                        self.prev_tray_mask_ref = tray_mask.copy()
                    else:
                        self.prev_tray_mask_ref = tray_mask.copy()
                        self.tray_stable_start_time = now
                        self.current_tray_logged = False
            else:
                self.prev_tray_mask_ref = None
                self.tray_stable_start_time = None
                self.current_tray_logged = False

            vis = frame.copy()
            vis = visualize_baguettes_and_tray(
                vis, (rx1, ry1, rx2, ry2),
                [itm["mask"] for itm in filtered_baguettes],
                tray_mask if tray_mask is not None else None,
                alpha=self.config["alpha"]
            )
            
            if measurement_draw_enabled and (do_height or do_width) and tray_mask is not None:
                vis = draw_measurement_lines(
                    vis, (rx1, ry1, rx2, ry2),
                    filtered_baguettes,
                    draw_height=do_height,
                    draw_width=do_width
                )

            del lb, roi, masks_roi
            if 'filtered_baguettes' in locals():
                del filtered_baguettes
            if 'tray_masks' in locals():
                del tray_masks
            
            return vis

        except Exception as e:
            print(f"Error in frame processing: {e}")
            return frame

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
            "tray_count": self.tray_counter,
            "config": self.config,
            "initialization_error": self.initialization_error,
            "is_stuck": self.is_stuck(),
            "last_activity": time.time() - self.last_activity_time,
        }

