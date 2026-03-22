import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = ["car", "van", "truck", "bus"]

def nms(boxes, scores, iou_threshold=0.5):
    indices = scores.argsort()[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        rest = indices[1:]
        ious = compute_iou(boxes[current], boxes[rest])

        indices = rest[ious < iou_threshold]

    return keep


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])

    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)

    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

    union = area1 + area2 - inter
    return inter / (union + 1e-6)



# Load model
#model = YOLO("yolov8s.pt")  # or better: aerial-trained weights
model = YOLO("yolov8l-visdrone.pt")  # or better: aerial-trained weights

# Parameters
#TILE_SIZE = 640
#OVERLAP = 0.25   # 25% overlap helps avoid border misses
#CONF_THRES = 0.005
#IOU_NMS = 0.5

TILE_SIZE = 512        # smaller = better for small cars
OVERLAP = 0.3         # higher overlap for dense parking
CONF_THRES = 0.15
IOU_NMS = 0.45

# Note: Changed to test_capture.jpg as image.jpg doesn't exist
image = cv2.imread("test_capture.jpg")
if image is None:
    print("Error: test_capture.jpg not found.")
    exit(1)

image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

H, W = image.shape[:2]

# Generate tiles
stride = int(TILE_SIZE * (1 - OVERLAP))
tiles = []

for y in range(0, H, stride):
    for x in range(0, W, stride):
        tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
        tiles.append((tile, x, y))

# Collect detections
all_boxes = []
all_scores = []

for tile, x_offset, y_offset in tiles:
    results = model(tile, conf=CONF_THRES, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        # Convert to global coords
        x1 += x_offset
        x2 += x_offset
        y1 += y_offset
        y2 += y_offset

        all_boxes.append([x1, y1, x2, y2])
        all_scores.append(conf)

all_boxes = np.array(all_boxes)
all_scores = np.array(all_scores)

keep = nms(all_boxes, all_scores, IOU_NMS)
final_boxes = all_boxes[keep]

for (x1, y1, x2, y2) in final_boxes:
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

cv2.imwrite("detections.jpg", image)

print("Total cars detected:", len(final_boxes))

# Draw all boxes on the image
for box, score in zip(all_boxes, all_scores):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{score:.2f}", (x1, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("yo_result.jpg", image)
print(f"Detected {len(all_boxes)} boxes. Saved to yo_result.jpg")

