
from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import cv2
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
from onnx_to_tensorrt import get_engine

TRT_LOGGER = trt.Logger()


def prepare_data(img_path, input_size):
    img_raw = cv2.imread(img_path)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32)
    img /= 255.
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    # add this line, or cuda.memcpy_htod_async raises 'ValueError: ndarray is not contiguous'
    img = np.array(img, dtype=np.float32, order='C')
    return img_raw, img


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'test.onnx'
    engine_file_path = "test.trt"
    # Download a dog image and save it to the following file path:
    input_image_path = '../tmp/201905165_daytime-head_guobolu2_1_车头_SUV_白色.jpg'

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_HW = (224, 224)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    img_raw, img = prepare_data(input_image_path, input_resolution_HW)
    # Store the shape of the original input image in WH format, we will need it for later
    ih, iw = img_raw.shape[:2]
    shape_orig_WH = (iw, ih)
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = img
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.

    #print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))

if __name__ == '__main__':
    main()
