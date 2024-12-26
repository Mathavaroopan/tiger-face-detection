import cv2
from ultralytics import YOLO

# Load the YOLOv8-Face model
model = YOLO("D:\Face-Detection\yolov8-testing\yolov8n-face.pt")  # Replace with the correct model path

# Open the video file
video_path = r"D:\Face-Detection\yolov8-testing\tigers-video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter(
    "output_video.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame)

    # Annotate detections on the frame
    for box in results[0].boxes.data:
        # Extract box coordinates and confidence
        x1, y1, x2, y2, conf = box[:5].cpu().numpy()
        
        # Filter low-confidence detections
        if conf > 0.5:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
