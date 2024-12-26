import torch
import cv2
from yolov5 import YOLOv5  # Assuming YOLOv5 is installed locally

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_faces(image_path):
    results = model(image_path)
    return results.xyxy[0].numpy()  # Bounding boxes

def crop_faces(image, face_coords):
    faces = []
    for (x_min, y_min, x_max, y_max, _, _) in face_coords:
        cropped_face = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        faces.append(cropped_face)
    return faces

# Load your group image
image_path = r'D:\Face-Detection\group.jpg'
image = cv2.imread(image_path)
face_coords = detect_faces(image_path)

faces = crop_faces(image, face_coords)

# Display cropped faces for inspection
for idx, face in enumerate(faces):
    cv2.imshow(f"Tiger {idx+1}", face)
cv2.waitKey(0)
cv2.destroyAllWindows()
