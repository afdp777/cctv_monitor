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
import os, time, json
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

ort_session = None
input_name = None
detection_time = 0
detected_class_id = -1
drawing = False
line_pt1 = line_pt2 = None
aspect_ratio = 1


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

def draw_line(event, x, y, flags, param):
    global drawing, line_pt1, line_pt2, aspect_ratio
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_pt1 = (x, y)
        line_pt2 = None
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        line_pt2 = (x, y)
        # save the points representing the line
        with open("line.json", "w") as f:
            json.dump({"line_pt1": line_pt1, "line_pt2": line_pt2}, f)

def box_intersects_line(bx1, by1, bx2, by2, line_pt1, line_pt2):
    # Normalize in case coords are not ordered
    x_min, y_min = min(bx1, bx2), min(by1, by2)
    x_max, y_max = max(bx1, bx2), max(by1, by2)

    # Convert to rect (x, y, w, h)
    rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    retval, _, _ = cv2.clipLine(rect, line_pt1, line_pt2)
    return retval

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
    global COCO_CLASSES, detection_time, detected_class_id, line_pt1, line_pt2, aspect_ratio
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
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        conf = confidences[i]
        class_id = class_ids[i]
        # person of interest
        if COCO_CLASSES[class_id] in ['person', 'dog', 'cat']: # class_id == 0: # "person"
            label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 2)
            # issue alarm logs when there's a defined monitoring line
            if line_pt1 is not None and line_pt2 is not None:
                # the line was drawn based on dimension of window, so we scale it to the dimension of 'img'.
                point1 = (int(line_pt1[0]*aspect_ratio), int(line_pt1[1]*aspect_ratio))
                point2 = (int(line_pt2[0]*aspect_ratio), int(line_pt2[1]*aspect_ratio))
                # log only when bounding box intersects with monitoring line
                intersects = box_intersects_line(x1, y1, x2, y2, point1, point2)
                # limit logging to console to every 30 seconds
                if intersects and (detected_class_id != class_id or int(time.time() - detection_time) >= 30):
                    detection_time = time.time()
                    detected_class_id = class_id
                    print(f"{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(detection_time))}: {COCO_CLASSES[class_id]} detected")
    return img


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    global aspect_ratio
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    aspect_ratio = dim[0] / dim[1]
    return cv2.resize(image, dim, interpolation=inter)


def main():
    global ort_session, input_name, line_pt1, line_pt2
    # download the ultralytics yolo model
    export_yolo_to_onnx()
    # load the line points
    try:
        with open("line.json", "r") as f:
            data = json.load(f)
            line_pt1 = tuple(data["line_pt1"])
            line_pt2 = tuple(data["line_pt2"])
    except Exception as e:
        print(f"Error loading 'line.json'")

    # Run the inference on GPU (CUDA and OpenVINO were untested 'coz I don't have them)
    providers = ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'DmlExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model, providers=providers)
    # Get the input name from the ONNX model
    input_name = ort_session.get_inputs()[0].name
    # Get available provider
    provider = [p for p in providers if p in ort_session.get_providers()]
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    target_fps = 12

    print(f"\nPerforming YOLO detection ", end="")
    if provider:
        print(f"using GPU via {provider[0]}")
    else:
        print("using CPU")
    print(f"YOLO model: {yolo_model}\nTarget fps: {target_fps}")
    print("Detection logs with timestamp follows here.\nCorrelate with timestamp of recorded CCTV video.\n*****BEGIN LOG*****")

    window_name = 'YOLO on MJPEG Stream using GPU'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_line)

    while True:
        # Skip frames to catch up
        skip_count = max(0, fps - target_fps)
        for _ in range(skip_count):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print(f"Frame capture failed.")
            break

        # resize, normalize, convert to CHW
        input_img = preprocess_image(frame)        
        # Run YOLO detection on the input image
        results = process_image(input_img)
        # extract detections, draw boxes
        result_img = postprocess_image(results, frame, 0.25)

        # resize to something that's easily viewable onscreen
        result_img = resize_with_aspect_ratio(result_img, width=720)
        # Show the frame rate
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = int(frame_count / elapsed)
            if elapsed >= 60:
                frame_count = 0
                start_time = time.time()
            cv2.putText(result_img, f"{fps}fps", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 1)    
        # Draw the line
        if line_pt1 is not None and line_pt2 is not None:
            cv2.line(result_img, line_pt1, line_pt2, (0, 255, 255), 2)
        # Display the resulting image
        cv2.imshow(window_name, result_img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()