# The MIT License (MIT)

# Copyright (c) 2025 Anthony Dela Paz, https://github.com/afdp777/cctv_monitor
# Based on RyzenAI-SW, https://github.com/amd/RyzenAI-SW/blob/main/tutorial/object_detection/

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
import cv2
import os, time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

# URL of the MJPEG stream
stream_url = "rtsp://192.168.1.23:554/live/ch01_1"

# Create a VideoCapture object to read from the MJPEG stream
cap = cv2.VideoCapture(stream_url)
# Use this if you want to capture from the webcam
#cap = cv2.VideoCapture(0)

yolo_model = "yolo11n.pt"
onnx_model = f"{yolo_model.split('.')[0]}.onnx"
COCO_CLASSES = []

# Run the inference on GPU
providers = ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider']
ort_session = ort.InferenceSession(onnx_model, providers=providers)

# Get the input name from the ONNX model
input_name = ort_session.get_inputs()[0].name


def extract_coco_classes_from_ultralytics_yolo(model):
    coco_classes_dict = model.model.names
    coco_class_names = list(coco_classes_dict.values())
    return coco_class_names

# Export the .pt file to .onnx
def export_yolo_to_onnx():
    global COCO_CLASSES
    model = YOLO(yolo_model)
    COCO_CLASSES = extract_coco_classes_from_ultralytics_yolo(model)
    if not os.path.exists(onnx_model):
        #print("Number of classes:", model.model.nc)
        model.export(format="onnx", opset=17)


# Preprocessing: resize, normalize, convert to CHW
def preprocess_image(img, img_size=640):
    #orig = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img #, orig


# Perform YOLO detection
def process_image(input_img):
    global ort_session, input_name
    return (ort_session.run(None, {input_name: input_img}))


# Postprocessing: extract detections, draw boxes
def postprocess_image(outputs, img, conf_thres=0.4):
    global COCO_CLASSES
    predictions = np.transpose(np.squeeze(outputs[0]))
    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        x_center, y_center, width, height = pred[0:4]
        class_scores = pred[4:]

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < conf_thres:
            continue

        # Convert xywh (center) to xyxy (corners)
        x1 = int((x_center - width / 2) * img.shape[1] / 640)
        y1 = int((y_center - height / 2) * img.shape[0] / 640)
        x2 = int((x_center + width / 2) * img.shape[1] / 640)
        y2 = int((y_center + height / 2) * img.shape[0] / 640)

        # boxes.append((x1, y1, x2, y2, confidence, class_id))
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_threshold=0.5)

    # Draw boxes
    # for (x1, y1, x2, y2, conf, class_id) in boxes:
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        conf = confidences[i]
        class_id = class_ids[i]
        # person of interest
        if class_id == 0: # "person"
            label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 2)
    return img


def main():
    global ort_session, input_name
    # download the ultralytics yolo model
    export_yolo_to_onnx()
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    target_fps = 20

    print(f"\nPerforming YOLO detection ", end="")
    provider = [p for p in providers if p in ort_session.get_providers()]
    
    if provider:
        print(f"using GPU via {provider[0]}")
    else:
        print("using CPU")
    print(f"YOLO model: {yolo_model}\nTarget fps: {target_fps}")

    while True:
        # Skip frames to catch up
        skip_count = max(0, fps - target_fps)
        for _ in range(skip_count):
            print("skipped frames")
            cap.read()

        ret, frame = cap.read()
        if not ret:
            break

        # resize, normalize, convert to CHW
        input_img = preprocess_image(frame)        
        # Run YOLO detection on the input image
        results = process_image(input_img)
        #results = ort_session.run(None, {input_name: input_img})
        # extract detections, draw boxes
        result_img = postprocess_image(results, frame, 0.3)
        # Show the frame rate
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = int(frame_count / elapsed)
            cv2.putText(result_img, f"{fps}fps", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 1)
        # Display the resulting image
        cv2.imshow('YOLO on MJPEG Stream using GPU', result_img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()