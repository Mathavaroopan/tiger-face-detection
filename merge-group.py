import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 trained model for tiger face detection
model = YOLO(r"D:/Face-Detection/yolov8-testing/saved_models/tiger_face_model.pt")  # Correct path to your trained model

# Load YOLOv5 model (for detecting multiple tigers in the group image)
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_faces(image_path):
    # Detect all objects (including tigers) in the image
    results = yolov5_model(image_path)
    return results.xyxy[0].numpy()  # Return bounding boxes (x1, y1, x2, y2, confidence, class)

def crop_faces(image, face_coords):
    # Crop the faces from the image based on detected bounding boxes
    faces = []
    for (x_min, y_min, x_max, y_max, _, _) in face_coords:
        cropped_face = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        faces.append((cropped_face, (x_min, y_min, x_max, y_max)))  # Store crop and original box coordinates
    return faces

# Load your group image
image_path = r'D:/Face-Detection/group.jpg'  # Correct path to your group image
image = cv2.imread(image_path)

# Step 1: Detect tigers in the group image
face_coords = detect_faces(image_path)

# Step 2: Crop the detected tiger faces
faces = crop_faces(image, face_coords)

# Step 3: Process each cropped face using YOLOv8 for further detection
for idx, (face, (x_min, y_min, x_max, y_max)) in enumerate(faces):
    # Run YOLOv8 inference on each cropped face
    results = model(face)
    
    # Step 4: Initialize a variable to track the highest confidence box
    highest_confidence = 0
    best_box = None

    # Step 5: Loop through all detected boxes for the current face
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy()  # Get bounding box coordinates (x1, y1, x2, y2)
        conf = result.conf.cpu().numpy()[0]   # Get confidence score
        
        # Step 6: Filter out small boxes based on area (too small to be a tiger face)
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        # Only keep boxes that have a sufficient area (e.g., > 500 pixels)
        if box_area > 500 and conf > highest_confidence:
            highest_confidence = conf
            best_box = box

    # Step 7: If a valid box is found, draw it on the image
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(face, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
        label = f"Tiger Face {highest_confidence:.2f}"
        cv2.putText(face, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the cropped face with the highest confidence bounding box
    cv2.imshow(f"Tiger {idx + 1}", face)

cv2.waitKey(0)
cv2.destroyAllWindows()
