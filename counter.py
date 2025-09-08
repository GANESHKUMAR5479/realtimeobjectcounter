from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")  # Pre-trained on COCO dataset

# Load traffic video
video_path = "fff.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)

    
    class_names = model.names

    
    vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
    vehicle_count = 0

    
    for class_id in class_ids:
        class_name = class_names[class_id]
        if class_name in vehicle_classes:
            vehicle_count += 1

    
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Vehicles Detected: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    
    cv2.imshow("Traffic Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()
