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
DEFAULT_TOPK = 100
DEFAULT_CANVAS = 640

CLASSES = ('bad','baguette','good','hole')
LABEL_BAD, LABEL_BAGUETTE, LABEL_GOOD, LABEL_HOLE = 0, 1, 2, 3

COLOR_RED   = (0,   0, 255)
COLOR_GREEN = (0, 255,   0)
COLOR_BLUE  = (255, 0,   0)
COLOR_BLACK = (0,   0,   0)
COLOR_WHITE = (255, 255, 255)

CLASS_TO_COLOR = {
    LABEL_BAD:  COLOR_RED,
    LABEL_GOOD: COLOR_GREEN,
    LABEL_HOLE: COLOR_BLUE,
}
DRAW_PRIORITY = {LABEL_GOOD: 0, LABEL_BAD: 1, LABEL_HOLE: 2}

def parse_button(s: str):
    if s is None: return set()
    s = s.strip()
    if s == "" or s == "[]": return set()
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    tokens = []
    for t in s.split(","):
        tt = t.strip().lower().strip("'\"")
        if tt:
            tokens.append(tt)
    valid = {"outer_diameter", "inner_diameter"}
    return set(t for t in tokens if t in valid)

def load_config(path: str):
    with open(path, "r") as f:
        cfg = json.load(f)
    roi = cfg.get("ROI", {})
    ref = cfg.get("reference", {})

    def fget(d, k, default=None):
        v = d.get(k, default)
        try:
            return float(v)
        except Exception:
            return default

    cx = fget(roi, "x")
    cy = fget(roi, "y")
    rw = fget(roi, "width")
    rh = fget(roi, "height")
    if None in (cx, cy, rw, rh):
        raise ValueError("ROI values (x,y,width,height) must be provided in the JSON.")

    px_len = fget(ref, "pixel_length")
    ac_len = fget(ref, "actual_length")
    unit = ref.get("unit", "mm")
    mm_per_px = None
    if px_len and ac_len and px_len > 0:
        mm_per_px = ac_len / px_len
    return cx, cy, rw, rh, mm_per_px, unit

def fmt_len(px_val: int, scale_units_per_px, unit_label: str):
    if scale_units_per_px is None:
        return f"{int(px_val)}"
    else:
        return f"{int(px_val)}px ({px_val*scale_units_per_px:.3f} {unit_label})"

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
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - top)  / scale, 0, out_h - 1)
    return out

def unletterbox_masks_roi(masks_u8, boxes_img, scale, left, top, out_w, out_h, max_workers=4):
    if masks_u8.size == 0:
        return masks_u8, 0.0, 0.0
    N, Hm, Wm = masks_u8.shape
    out_masks = np.zeros((N, out_h, out_w), dtype=np.uint8)

    x1 = np.clip(left + boxes_img[:, 0] * scale, 0, Wm).astype(np.int32)
    y1 = np.clip(top  + boxes_img[:, 1] * scale, 0, Hm).astype(np.int32)
    x2 = np.clip(left + boxes_img[:, 2] * scale, 0, Wm).astype(np.int32)
    y2 = np.clip(top  + boxes_img[:, 3] * scale, 0, Hm).astype(np.int32)

    dx1 = np.clip(boxes_img[:, 0], 0, out_w).astype(np.int32)
    dy1 = np.clip(boxes_img[:, 1], 0, out_h).astype(np.int32)
    dx2 = np.clip(boxes_img[:, 2], 0, out_w).astype(np.int32)
    dy2 = np.clip(boxes_img[:, 3], 0, out_h).astype(np.int32)

    def place(i):
        if x2[i] <= x1[i] or y2[i] <= y1[i] or dx2[i] <= dx1[i] or dy2[i] <= dy1[i]:
            return
        src = masks_u8[i, y1[i]:y2[i], x1[i]:x2[i]]
        if src.size == 0: return
        w_dst = int(dx2[i] - dx1[i]); h_dst = int(dy2[i] - dy1[i])
        dst_roi = cv2.resize(src, (w_dst, h_dst), interpolation=cv2.INTER_NEAREST)
        out_masks[i, dy1[i]:dy1[i]+h_dst, dx1[i]:dx1[i]+w_dst] = dst_roi

    if N > 3:
        with ThreadPoolExecutor(max_workers=min(max_workers, N)) as ex:
            list(ex.map(place, range(N)))
    else:
        for i in range(N): place(i)
    return out_masks, 0.0, 0.0

def blend_masks_classwise(base_bgr, masks, labels, alpha=0.45):
    vis = base_bgr.copy()
    if masks.size == 0: return vis
    draw = []
    for i, lab in enumerate(labels):
        if int(lab) == LABEL_BAGUETTE: continue
        draw.append((DRAW_PRIORITY.get(int(lab), 1), i))
    if not draw: return vis
    draw.sort(key=lambda x: x[0])
    for _, i in draw:
        lab = int(labels[i]); m = masks[i]
        if m.sum() == 0: continue
        x, y, w, h = cv2.boundingRect(m)
        if w == 0 or h == 0: continue
        roi = vis[y:y+h, x:x+w]; mroi = m[y:y+h, x:x+w]
        color = CLASS_TO_COLOR.get(lab)
        if color is None: continue
        color_roi = np.full_like(roi, color, dtype=np.uint8)
        blended = cv2.addWeighted(roi, 1.0 - alpha, color_roi, alpha, 0.0)
        cv2.copyTo(blended, mroi, roi)
    return vis

def compute_roi_rect(frame_w, frame_h, cx, cy, w, h):
    w = int(round(w)); h = int(round(h))
    x1 = int(round(cx - w / 2.0)); y1 = int(round(cy - h / 2.0))
    x1 = max(0, min(x1, frame_w - 1)); y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x1 + w, frame_w)); y2 = max(y1 + 1, min(y1 + h, frame_h))
    return x1, y1, x2, y2

def collect_good_items(labels_kept, masks_roi):
    goods = []
    for i, lab in enumerate(labels_kept):
        if int(lab) != LABEL_GOOD: continue
        m = masks_roi[i]
        if m is None or m.size == 0 or m.sum() == 0: continue
        x, y, w, h = cv2.boundingRect(m)
        if w == 0 or h == 0: continue
        cx = x + w // 2
        cy = y + h // 2
        goods.append({"i": i, "x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy, "area": int(w*h)})
    return goods

def collect_hole_items(labels_kept, masks_roi):
    holes = []
    for i, lab in enumerate(labels_kept):
        if int(lab) != LABEL_HOLE: continue
        m = masks_roi[i]
        if m is None or m.size == 0 or m.sum() == 0: continue
        x, y, w, h = cv2.boundingRect(m)
        if w == 0 or h == 0: continue
        cx = x + w // 2
        cy = y + h // 2
        holes.append({"i": i, "x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy, "area": int(w*h)})
    return holes

def pair_holes_to_goods(goods_sorted, holes):
    hole_heights = []
    for g in goods_sorted:
        gx1, gy1 = g["x"], g["y"]
        gx2, gy2 = g["x"] + g["w"], g["y"] + g["h"]
        candidates = [h for h in holes if (gx1 <= h["cx"] <= gx2 and gy1 <= h["cy"] <= gy2)]
        if candidates:
            best = max(candidates, key=lambda hh: hh["area"])
            hole_heights.append(best["h"])
        else:
            hole_heights.append(0)
    return hole_heights

def compute_good_union_mask(labels_kept, masks_roi, H, W):
    if masks_roi.size == 0: return np.zeros((H, W), dtype=np.uint8)
    union = np.zeros((H, W), dtype=np.uint8)
    for i, lab in enumerate(labels_kept):
        if int(lab) == LABEL_GOOD:
            union |= (masks_roi[i] > 0).astype(np.uint8)
    return union

def make_locked_slots_from_goods(goods, roi_h, min_band=8, band_scale=0.4):
    goods = sorted(goods, key=lambda g: g["cy"])
    slots = []
    for k, g in enumerate(goods, start=1):
        h = max(1, int(g["h"]))
        half_band = max(min_band, int(band_scale * h))
        lo = max(0, g["cy"] - half_band)
        hi = min(roi_h, g["cy"] + half_band)
        slots.append({
            "idx": k,
            "last_x": float(g["cx"]),
            "last_y": float(g["cy"]),
            "band_lo": int(lo),
            "band_hi": int(hi),
        })
    return slots

def track_locked_slots(current_goods, locked_slots, tol_y=40, x_forward_slack=6, smooth=0.6):
    if not locked_slots: return []
    used = set(); out = []
    for slot in sorted(locked_slots, key=lambda s: s["idx"]):
        best_j, best_dy = None, float("inf")
        for j, g in enumerate(current_goods):
            if j in used: continue
            dy = abs(g["cy"] - slot["last_y"])
            if dy <= tol_y and g["cx"] <= slot["last_x"] + x_forward_slack:
                if dy < best_dy:
                    best_dy, best_j = dy, j
        if best_j is not None:
            used.add(best_j)
            g = current_goods[best_j]
            out.append({"x": g["x"], "y": g["y"], "idx": slot["idx"]})
            slot["last_y"] = smooth * slot["last_y"] + (1.0 - smooth) * g["cy"]
            slot["last_x"] = min(slot["last_x"], float(g["cx"]))
    return out

def all_slots_fully_out_left_restricted(good_union_mask, locked_slots, left_exit_slack=2):
    if not locked_slots: return True
    H, W = good_union_mask.shape
    for s in locked_slots:
        lo = max(0, min(H, s["band_lo"]))
        hi = max(0, min(H, s["band_hi"]))
        if lo >= hi: return False
        limit = int(min(W, max(1, s["last_x"] + left_exit_slack)))
        band_left = good_union_mask[lo:hi, :limit]
        if np.any(band_left):
            return False
    return True

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
            self.start_evt = cuda.Event(); self.end_evt = cuda.Event()

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
            }""")
            self._k_thresh = self._thresh_mod.get_function("thresh_u8")
            self.dev_masks_u8 = None; self.u8_capacity = 0; self.mask_host_buf_u8 = None

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
            }""")
            self._k_roi = self._roi_mod.get_function("resize_thresh_roi")

            self.dev_rois_u8 = None; self.host_rois_u8 = None; self.roi_bytes_capacity = 0
            _ = cv2.resize(np.zeros((10,10), np.uint8), (20,20), interpolation=cv2.INTER_NEAREST)
            
            print("TRTSegmentor initialization complete")
            
        finally:
            pass

    def _setup_io(self):
        if self.trt10:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        def find_tensor(keys):
            for name in names:
                if any(k in name.lower() for k in keys): return name
            return None
        self.name_in     = find_tensor(['raw_input','input'])
        self.name_dets   = find_tensor(['det','boxes'])
        self.name_labels = find_tensor(['label','class'])
        self.name_masks  = find_tensor(['mask','seg'])
        if any(x is None for x in [self.name_in, self.name_dets, self.name_labels, self.name_masks]):
            raise RuntimeError(f"Could not locate required tensors: {names}")

        if self.trt10:
            self.dtype_in     = trt.nptype(self.engine.get_tensor_dtype(self.name_in))
            self.dtype_dets   = trt.nptype(self.engine.get_tensor_dtype(self.name_dets))
            self.dtype_labels = trt.nptype(self.engine.get_tensor_dtype(self.name_labels))
            self.dtype_masks  = trt.nptype(self.engine.get_tensor_dtype(self.name_masks))
        else:
            def gd(name):
                idx = self.engine.get_binding_index(name)
                return trt.nptype(self.engine.get_binding_dtype(idx))
            self.dtype_in, self.dtype_dets = gd(self.name_in), gd(self.name_dets)
            self.dtype_labels, self.dtype_masks = gd(self.name_labels), gd(self.name_masks)

        self.input_shape = (1, 640, 640, 3)
        self.max_det = 100
        self.mask_hw = (640, 640)
        self.dev_ptr, self.host_buf = {}, {}

        def alloc(name, shape, dtype):
            n = int(np.prod(shape)); nbytes = n * np.dtype(dtype).itemsize
            buf = cuda.pagelocked_empty(n, dtype) if nbytes >= PINNED_THRESHOLD_BYTES else np.empty(n, dtype=dtype)
            self.host_buf[name] = buf
            self.dev_ptr[name] = cuda.mem_alloc(nbytes)

        alloc(self.name_in, self.input_shape, self.dtype_in)
        alloc(self.name_dets, (1, self.max_det, 5), self.dtype_dets)
        alloc(self.name_labels, (1, self.max_det), self.dtype_labels)
        mask_bytes = int(np.prod((1, self.max_det, *self.mask_hw))) * np.dtype(self.dtype_masks).itemsize
        self.dev_ptr[self.name_masks] = cuda.mem_alloc(mask_bytes)

        if self.trt10:
            for name, ptr in self.dev_ptr.items():
                self.context.set_tensor_address(name, int(ptr))
        else:
            self.bindings = [None] * self.engine.num_bindings
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                self.bindings[i] = int(self.dev_ptr[name])

    def infer_optimized(self, img_uint8):
        np.copyto(self.host_buf[self.name_in], img_uint8.ravel())
        cuda.memcpy_htod_async(self.dev_ptr[self.name_in], self.host_buf[self.name_in], self.stream)
        self.start_evt.record(self.stream)
        if self.trt10: self.context.execute_async_v3(self.stream.handle)
        else:          self.context.execute_async_v2(self.bindings, self.stream.handle)
        self.end_evt.record(self.stream)
        cuda.memcpy_dtoh_async(self.host_buf[self.name_dets],   self.dev_ptr[self.name_dets],   self.stream)
        cuda.memcpy_dtoh_async(self.host_buf[self.name_labels], self.dev_ptr[self.name_labels], self.stream)
        self.stream.synchronize()

        dets_raw   = self.host_buf[self.name_dets].reshape(1, self.max_det, 5)[0]
        labels_raw = self.host_buf[self.name_labels].reshape(1, self.max_det)[0]
        boxes  = dets_raw[:, :4].astype(np.float32, copy=False)
        scores = dets_raw[:, 4].astype(np.float32, copy=False)
        labels = labels_raw.astype(np.int32, copy=False)
        return boxes, labels, scores

    def _ensure_u8_capacity(self, needed_masks):
        Hm, Wm = self.mask_hw
        max_reasonable_capacity = 200  # Prevent unlimited growth
        needed_capacity = min(needed_masks, max_reasonable_capacity)
        
        if needed_capacity <= getattr(self, "u8_capacity", 0): return
        
        # If current capacity is much larger than needed, shrink it
        current_cap = getattr(self, "u8_capacity", 0)
        if current_cap > needed_capacity * 4:  # Shrink if 4x larger than needed
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
        max_reasonable_bytes = 50 * 1024 * 1024  # 50MB max
        needed_bytes = min(needed_bytes, max_reasonable_bytes)
        
        if needed_bytes <= getattr(self, "roi_bytes_capacity", 0): return
        
        # If current capacity is much larger than needed, shrink it
        current_cap = getattr(self, "roi_bytes_capacity", 0)
        if current_cap > needed_bytes * 4:  # Shrink if 4x larger than needed
            needed_bytes = max(needed_bytes * 2, 1 << 20)
        else:
            needed_bytes = max(needed_bytes * 2, 1 << 20)
            
        needed_bytes = min(needed_bytes, max_reasonable_bytes)
        
        self.roi_bytes_capacity = needed_bytes
        if getattr(self, "dev_rois_u8", None) is not None: self.dev_rois_u8.free()
        self.dev_rois_u8 = cuda.mem_alloc(self.roi_bytes_capacity)
        self.host_rois_u8 = cuda.pagelocked_empty(self.roi_bytes_capacity, dtype=np.uint8)

    def copy_masks_optimized(self, indices, thr=0.5):
        if len(indices) == 0:
            return np.empty((0, *self.mask_hw), dtype=self.dtype_masks), {}
        Hm, Wm = self.mask_hw; n_pix = Hm * Wm
        base_addr = int(self.dev_ptr[self.name_masks]); itemsize = np.dtype(self.dtype_masks).itemsize
        slice_bytes = Hm * Wm * itemsize

        if self.dtype_masks == np.uint8:
            if getattr(self, "mask_host_buf", None) is None or getattr(self, "mask_host_capacity", 0) < len(indices):
                self.mask_host_capacity = max(len(indices) * 2, 32)
                self.mask_host_buf = cuda.pagelocked_empty((self.mask_host_capacity, Hm, Wm), dtype=np.uint8)
            for k, idx in enumerate(indices):
                src = int(base_addr + int(idx) * slice_bytes)
                dst = self.mask_host_buf[k].ravel()
                cuda.memcpy_dtoh_async(dst, src, stream=self.stream)
            self.stream.synchronize()
            return self.mask_host_buf[:len(indices)].copy(), {}

        self._ensure_u8_capacity(len(indices))
        threads = 256
        for k, idx in enumerate(indices):
            src_offset = int(idx) * n_pix
            dst_offset = k * n_pix
            grid = ((n_pix + threads - 1) // threads, 1, 1)
            self._k_thresh(self.dev_ptr[self.name_masks], self.dev_masks_u8,
                           np.int32(n_pix), np.int32(src_offset), np.int32(dst_offset), np.float32(thr),
                           block=(threads,1,1), grid=grid, stream=self.stream)
        self.stream.synchronize()
        cuda.memcpy_dtoh_async(self.mask_host_buf_u8[:len(indices)].ravel(), int(self.dev_masks_u8), stream=self.stream)
        self.stream.synchronize()
        return self.mask_host_buf_u8[:len(indices)].copy(), {}

    def copy_and_resize_masks_roi_gpu(self, indices, boxes_img, scale, left, top, out_w, out_h, thr=0.5):
        if len(indices) == 0:
            return np.empty((0, out_h, out_w), dtype=np.uint8), {}
        if self.dtype_masks != np.float32:
            raise RuntimeError("GPU ROI path expects float32 mask output from engine.")

        Hm, Wm = self.mask_hw; n_pix = Hm * Wm
        boxes_img = boxes_img.astype(np.float32, copy=False)
        sx = np.clip(left + boxes_img[:,0] * scale, 0, Wm).astype(np.int32)
        sy = np.clip(top  + boxes_img[:,1] * scale, 0, Hm).astype(np.int32)
        ex = np.clip(left + boxes_img[:,2] * scale, 0, Wm).astype(np.int32)
        ey = np.clip(top  + boxes_img[:,3] * scale, 0, Hm).astype(np.int32)
        sw = np.maximum(ex - sx, 0).astype(np.int32)
        sh = np.maximum(ey - sy, 0).astype(np.int32)
        dx = np.clip(boxes_img[:,0], 0, out_w).astype(np.int32)
        dy = np.clip(boxes_img[:,1], 0, out_h).astype(np.int32)
        ex2= np.clip(boxes_img[:,2], 0, out_w).astype(np.int32)
        ey2= np.clip(boxes_img[:,3], 0, out_h).astype(np.int32)
        dw = np.maximum(ex2 - dx, 0).astype(np.int32)
        dh = np.maximum(ey2 - dy, 0).astype(np.int32)

        sizes = (dw * dh).astype(np.int64)
        valid = (sw > 0) & (sh > 0) & (dw > 0) & (dh > 0)
        sizes[~valid] = 0
        offsets = np.zeros(len(indices), dtype=np.int64)
        if len(indices) > 0: np.cumsum(sizes[:-1], out=offsets[1:])
        total_elems = int(offsets[-1] + sizes[-1]) if len(indices) > 0 else 0
        self._ensure_roi_bytes(max(total_elems, 1))

        masks_full = np.zeros((len(indices), out_h, out_w), dtype=np.uint8)
        block = (16,16,1)
        for i, idx in enumerate(indices):
            if sizes[i] == 0: continue
            src_offset = int(idx) * n_pix
            dst_offset = int(offsets[i])
            grid = ((int(dw[i])+block[0]-1)//block[0], (int(dh[i])+block[1]-1)//block[1], 1)
            self._k_roi(self.dev_ptr[self.name_masks],
                        np.int32(Hm), np.int32(Wm),
                        np.int32(src_offset),
                        np.int32(int(sx[i])), np.int32(int(sy[i])),
                        np.int32(int(sw[i])), np.int32(int(sh[i])),
                        self.dev_rois_u8,
                        np.int32(dst_offset),
                        np.int32(int(dw[i])), np.int32(int(dh[i])),
                        np.float32(thr),
                        block=block, grid=grid, stream=self.stream)
        self.stream.synchronize()
        if total_elems > 0:
            cuda.memcpy_dtoh_async(self.host_rois_u8[:total_elems], int(self.dev_rois_u8), stream=self.stream)
        self.stream.synchronize()
        base = self.host_rois_u8
        for i in range(len(indices)):
            if sizes[i] == 0: continue
            count = int(sizes[i]); off = int(offsets[i])
            roi = np.frombuffer(base, dtype=np.uint8, count=count, offset=off).reshape(int(dh[i]), int(dw[i]))
            masks_full[i, dy[i]:dy[i]+int(dh[i]), dx[i]:dx[i]+int(dw[i])] = roi
        return masks_full, {}

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
        
        self.state = "seek"
        self.locked_slots = []
        self.gap_t0 = None
        self.roi_config = None
        self.button_set = set()

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
            "score_threshold": config.get("score_threshold", 0.5),
            "mask_threshold": config.get("mask_threshold", 0.4),
            "canvas_size": config.get("canvas_size", 640),
            "alpha": config.get("alpha", 0.45),
            "target_fps": config.get("target_fps", 30.0),
            "mqtt_topic": config.get("mqtt_topic", "detection/results"),
            "csv_output_dir": config.get("csv_output_dir", None),
            "gap_delay": config.get("gap_delay", 1.0),
            "min_index_count": config.get("min_index_count", 1),
            "match_tol_y": config.get("match_tol_y", 50.0),
            "x_forward_slack": config.get("x_forward_slack", 6.0),
            "left_exit_slack": config.get("left_exit_slack", 2.0),
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
            roi_cx, roi_cy, roi_w, roi_h, units_per_px, units_label = load_config(self.config["config_path"])
            self.roi_config = {
                "cx": roi_cx, "cy": roi_cy, "w": roi_w, "h": roi_h,
                "units_per_px": units_per_px, "units_label": units_label
            }
            self.button_set = parse_button(self.config["button"])
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

    def _add_csv_data(self, row_number, doughnut_number, status, outer_diameter, inner_diameter):
        csv_row = {
            'row_number': row_number,
            'doughnut_number': doughnut_number,
            'status': status,
            'outer_diameter': outer_diameter,
            'inner_diameter': inner_diameter
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
            csv_filename = f"doughnut_measurements_{timestamp_str}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)

            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['row_number', 'doughnut_number', 'status', 'outer_diameter', 'inner_diameter']
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
        self.row_counter = 0
        self.inference_active = False
        self.initialization_error = None
        self.last_activity_time = time.time()
        
        self.state = "seek"
        self.locked_slots = []
        self.gap_t0 = None

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

            x1, y1, x2, y2 = compute_roi_rect(width, height, 
                                             self.roi_config["cx"], self.roi_config["cy"],
                                             self.roi_config["w"], self.roi_config["h"])
            roi_w = x2 - x1
            roi_h = y2 - y1
            
            enable_outer = "outer_diameter" in self.button_set
            enable_inner = "inner_diameter" in self.button_set

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
                        frame, current_video_time_seconds, x1, y1, x2, y2, roi_w, roi_h,
                        enable_outer, enable_inner, frame_idx, frame_count
                    )

                    fps_counter.update()
                    self.current_fps = fps_counter.get_fps()

                    self._emit_frame(
                        vis,
                        {
                            "timestamp": current_video_time_seconds,
                            "fps": self.current_fps,
                            "frame_idx": frame_idx,
                            "row_number": self.row_counter,
                            "state": self.state,
                        },
                    )

                    frame_idx += 1

                    sleep_t = frame_interval - (time.perf_counter() - frame_t0)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

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

    def _process_frame(self, frame, current_video_time_seconds, x1, y1, x2, y2, roi_w, roi_h, 
                      enable_outer, enable_inner, frame_idx, frame_count):
        try:
            vis = frame.copy()
            roi = vis[y1:y2, x1:x2].copy()

            lb, scale, left, top = letterbox(roi, size=self.config["canvas_size"], pad_val=114)
            boxes, labels, scores = self.seg.infer_optimized(lb)

            keep = scores >= self.config["score_threshold"]
            vis_roi = roi
            current_goods, current_holes = [], []
            current_bad_count = 0
            good_union = np.zeros((roi_h, roi_w), dtype=np.uint8)

            if np.any(keep):
                boxes_kept = boxes[keep]
                labels_kept = labels[keep]
                keep_idx = np.flatnonzero(keep)
                boxes_roi = unletterbox_boxes(boxes_kept, scale, left, top, roi_w, roi_h)

                if self.config["gpu_roi"] and self.seg.dtype_masks == np.float32:
                    masks_roi, _ = self.seg.copy_and_resize_masks_roi_gpu(
                        keep_idx, boxes_roi, scale, left, top, roi_w, roi_h, thr=self.config["mask_threshold"]
                    )
                else:
                    masks_fp, _ = self.seg.copy_masks_optimized(keep_idx, thr=self.config["mask_threshold"])
                    masks_u8 = (masks_fp.astype(np.uint8) if self.config["engine_binary_masks"]
                                else ((masks_fp > self.config["mask_threshold"]) * 255).astype(np.uint8) if masks_fp.size > 0
                                else masks_fp.astype(np.uint8))
                    masks_roi, _, _ = unletterbox_masks_roi(
                        masks_u8, boxes_roi, scale, left, top, roi_w, roi_h, max_workers=self.config["workers"]
                    )

                vis_roi = blend_masks_classwise(roi, masks_roi, labels_kept, alpha=self.config["alpha"])

                current_goods = collect_good_items(labels_kept, masks_roi)
                if enable_inner:
                    current_holes = collect_hole_items(labels_kept, masks_roi)
                current_bad_count = sum(1 for i, lab in enumerate(labels_kept)
                                        if int(lab) == LABEL_BAD and masks_roi[i].sum() > 0)
                good_union = compute_good_union_mask(labels_kept, masks_roi, roi_h, roi_w)

                if enable_outer:
                    for g in current_goods:
                        y_mid = g["y"] + g["h"] // 2
                        cv2.line(vis_roi, (g["x"], y_mid), (g["x"] + g["w"], y_mid), COLOR_WHITE, 2)
                if enable_inner and current_holes:
                    for h in current_holes:
                        y_mid = h["y"] + h["h"] // 2
                        cv2.line(vis_roi, (h["x"], y_mid), (h["x"] + h["w"], y_mid), COLOR_BLACK, 2)

                self._update_state_machine(current_goods, current_holes, good_union, current_bad_count, vis_roi)
                vis[y1:y2, x1:x2] = vis_roi

            else:
                self._update_state_machine(current_goods, current_holes, good_union, current_bad_count, vis_roi)

            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BLACK, 2)
            
            return vis

        except Exception as e:
            print(f"Error in frame processing: {e}")
            return frame

    def _update_state_machine(self, current_goods, current_holes, good_union, current_bad_count, vis_roi):
        if self.state == "seek":
            if len(current_goods) >= self.config["min_index_count"]:
                goods_sorted = sorted(current_goods, key=lambda g: g["cy"])
                self.locked_slots = make_locked_slots_from_goods(goods_sorted, vis_roi.shape[0])
                self.row_counter += 1

                print(f"row {self.row_counter}")
                print(f" good:{len(goods_sorted)} bad:{current_bad_count}")
                hole_heights = pair_holes_to_goods(goods_sorted, current_holes) if "inner_diameter" in self.button_set else [0]*len(goods_sorted)
                
                for k, g in enumerate(goods_sorted, start=1):
                    outer_diam = fmt_len(int(g['h']), self.roi_config["units_per_px"], self.roi_config["units_label"]) if "outer_diameter" in self.button_set else None
                    inner_diam = fmt_len(int(hole_heights[k-1]) if hole_heights[k-1] else 0, self.roi_config["units_per_px"], self.roi_config["units_label"]) if "inner_diameter" in self.button_set else None
                    
                    parts = []
                    if outer_diam: parts.append(f"d{k}-diameter:{outer_diam}")
                    if inner_diam: parts.append(f"d{k}-hole:{inner_diam}")
                    if parts: print(" " + "  ".join(parts))
                    
                    status = "good"
                    self._add_csv_data(self.row_counter, k, status, outer_diam, inner_diam)

                self.state = "locked"
                self._draw_annotations(current_goods, vis_roi)

        elif self.state == "locked":
            self._draw_annotations(current_goods, vis_roi)

            if all_slots_fully_out_left_restricted(good_union, self.locked_slots,
                                                   left_exit_slack=self.config["left_exit_slack"]):
                self.state = "gap"
                self.gap_t0 = time.perf_counter()
                self.locked_slots = []

        elif self.state == "gap":
            if (time.perf_counter() - self.gap_t0) >= self.config["gap_delay"]:
                if len(current_goods) >= self.config["min_index_count"]:
                    goods_sorted = sorted(current_goods, key=lambda g: g["cy"])
                    self.locked_slots = make_locked_slots_from_goods(goods_sorted, vis_roi.shape[0])
                    self.row_counter += 1

                    print(f"row {self.row_counter}")
                    print(f" good:{len(goods_sorted)} bad:{current_bad_count}")
                    hole_heights = pair_holes_to_goods(goods_sorted, current_holes) if "inner_diameter" in self.button_set else [0]*len(goods_sorted)
                    
                    for k, g in enumerate(goods_sorted, start=1):
                        outer_diam = fmt_len(int(g['h']), self.roi_config["units_per_px"], self.roi_config["units_label"]) if "outer_diameter" in self.button_set else None
                        inner_diam = fmt_len(int(hole_heights[k-1]) if hole_heights[k-1] else 0, self.roi_config["units_per_px"], self.roi_config["units_label"]) if "inner_diameter" in self.button_set else None
                        
                        parts = []
                        if outer_diam: parts.append(f"d{k}-diameter:{outer_diam}")
                        if inner_diam: parts.append(f"d{k}-hole:{inner_diam}")
                        if parts: print(" " + "  ".join(parts))
                        
                        status = "good"
                        self._add_csv_data(self.row_counter, k, status, outer_diam, inner_diam)

                    self.state = "locked"
                    self.gap_t0 = None
                    self._draw_annotations(current_goods, vis_roi)

    def _draw_annotations(self, current_goods, vis_roi):
        ann = track_locked_slots(current_goods, self.locked_slots,
                                 tol_y=self.config["match_tol_y"],
                                 x_forward_slack=self.config["x_forward_slack"])
        for p in ann:
            cv2.putText(vis_roi, str(p["idx"]), (p["x"], max(0, p["y"]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis_roi, str(p["idx"]), (p["x"], max(0, p["y"]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

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
            "state": self.state,
        }

