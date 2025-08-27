import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT is not installed. Please install TensorRT and try again.")

import cv2


class TensorRTDetector:
    def __init__(self, engine_path, max_detections=1000):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.max_detections = max_detections
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def load_engine(self):
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding))
                * self.engine.max_batch_size
            )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def detect_raw_frame(self, frame, score_threshold=0.5):
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)

        np.copyto(self.inputs[0]["host"], frame.ravel())

        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        output = self.outputs[0]["host"]
        num_detections = int(output[0])

        detections = []
        for i in range(1, min(num_detections + 1, self.max_detections)):
            detection = output[i * 6 : (i + 1) * 6]
            if detection[5] < score_threshold:
                continue

            det = Detection(
                bbox=[detection[0], detection[1], detection[2], detection[3]],
                label_id=int(detection[4]),
                score=float(detection[5]),
            )
            detections.append(det)

        return detections


class Detection:
    def __init__(self, bbox, label_id, score):
        self.bbox = bbox
        self.label_id = label_id
        self.score = score
