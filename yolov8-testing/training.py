import cv2
from ultralytics import YOLO

# Step 1: Load the trained model
model = YOLO(r"D:/Face-Detection/yolov8-testing/saved_models/tiger_face_model.pt")  # Use the correct path to your saved model

# Step 2: Open the video
video_path = r"D:/Face-Detection/yolov8-testing/tigers-video.mp4"  # Path to your video file
cap = cv2.VideoCapture(video_path)

# Step 3: Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Step 4: Create a VideoWriter object to save the output video
out = cv2.VideoWriter(
    "output_tiger_faces.avi",  # Output video file name
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Step 5: Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 6: Run YOLOv8 inference on the frame
    results = model(frame)

    # Step 7: Annotate the frame with bounding boxes
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # Get bounding box coordinates (x1, y1, x2, y2)
        conf = result.conf.cpu().numpy()[0]   # Get confidence score as a scalar value

        # Step 8: Filter out detections that are too large (i.e., covering the whole body)
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        box_height = y2 - y1
        aspect_ratio = box_width / box_height

        if conf > 0.5 and aspect_ratio < 2:  # Adjust the aspect_ratio threshold if needed
            # Draw a rectangle around the detected tiger face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Tiger Face {conf:.2f}"  # Format the confidence score as a string
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Step 9: Write the annotated frame to the output video
    out.write(frame)

    # Optionally, display the frame
    cv2.imshow("Tiger Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the display
        break

# Step 10: Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output video saved.")
