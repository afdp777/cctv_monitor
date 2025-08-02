from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model (you can change to 'yolov8s.pt' for better accuracy)
model = YOLO('yolo11n.pt')

# URL of the MJPEG stream
stream_url = "rtsp://192.168.1.23:554/live/ch00_0"

# Create a VideoCapture object and read from the MJPEG stream
cap = cv2.VideoCapture(stream_url)


def main():
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    target_fps = 60
        
    while True:
        # Skip frames to catch up
        skip_count = max(0, fps - target_fps)
        for _ in range(skip_count):
            print("skipped frames")
            cap.read()

        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection on the frame
        results = model(frame, verbose=False)

        # Draw only 'person' detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if True: # results[0].names[class_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{results[0].names[class_id]}: {box.conf[0]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame rate
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = int(frame_count / elapsed)
            cv2.putText(frame, f"{fps}fps", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 250), 1)
        # Show the frame
        cv2.imshow('YOLO on MJPEG Stream using CPU', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()