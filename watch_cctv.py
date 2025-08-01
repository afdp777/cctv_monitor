import onnxruntime as ort
import numpy as np
import cv2
import ultralytics # for the yolo models

# Load COCO class labels
COCO_CLASSES = [  # 80 classes
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Preprocessing: resize, normalize, convert to CHW
def preprocess_image(img, img_size=640):
    orig = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img, orig

# Postprocessing: extract detections, draw boxes
def postprocess(outputs, img, conf_thres=0.4):
    predictions = np.transpose(np.squeeze(outputs[0]))
    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        x_center, y_center, width, height = pred[0:4]
        class_scores = pred[4:]  # 80 class scores

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
        label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 250, 250), 1)
    return img

# URL of the MJPEG stream
stream_url = "rtsp://192.168.1.23:554/live/ch01_1"

# Create a VideoCapture object and read from the MJPEG stream
cap = cv2.VideoCapture(stream_url)

def main():
    onnx_model = 'yolov8m.onnx'
    # Run the inference on GPU
    ort_session = ort.InferenceSession(onnx_model, providers=['DmlExecutionProvider'])

    # Get the input name from the ONNX model
    input_name = ort_session.get_inputs()[0].name

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # pre-process the input frame
        input_img, original = preprocess_image(frame)        
        # Run YOLO detection on the input image
        results = ort_session.run(None, {input_name: input_img})
        # post-process the results
        result_img = postprocess(results, original, 0.5)
        # Display the resulting image
        cv2.imshow('YOLO on MJPEG Stream', result_img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()