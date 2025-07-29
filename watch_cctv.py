from ultralytics import YOLO
import cv2

# Load YOLOv8 model (change to 'yolov8s.pt' for better accuracy)
model = YOLO('yolo11n.pt')
# URL of the MJPEG stream
stream_url = "rtsp://192.168.1.23:554/live/ch01_1"

# Create a VideoCapture object and read from the MJPEG stream
cap = cv2.VideoCapture(stream_url)

while True:
    # Read frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame, verbose=False)
    # Draw only 'person' detections
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        tag_string = results[0].names[class_id]
        if tag_string == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, tag_string, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO on MJPEG Stream', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()