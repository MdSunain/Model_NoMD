from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")


# Open video file
cap = cv2.VideoCapture("videos/traffic.mp4")

if not cap.isOpened():
    print("Cannot open video!")

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 360))   # Reduce HD to 360p
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Annotated frame (YOLO returns boxes drawn)
    annotated = results[0].plot()



    # Vehicle counting
    count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls in [2, 3, 5, 7]:  
            # 2=car, 3=motorcycle, 5=bus, 7=truck
            count += 1

    # Display count
    cv2.putText(
        annotated, 
        f"Vehicles: {count}", 
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    # Show output
    cv2.imshow("Vehicle Detection Sample", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
