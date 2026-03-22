import cv2
import numpy as np
import json
import os

class ParkingDetector:
    def __init__(self, config_path="config.json", edge_thresh=0.16, std_thresh=45.0, use_yolo=False, model_path="yolov8l-visdrone.pt"):
        self.config_path = config_path
        self.config_data = {}
        self.parking_slots = self.load_config()
        self.edge_thresh = edge_thresh
        self.std_thresh = std_thresh
        self.use_yolo = use_yolo

        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.vehicle_classes = ["car", "van", "truck", "bus"]
            except ImportError:
                print("Error: ultralytics is not installed. Falling back to simple heuristic.")
                self.use_yolo = False

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
                return self.config_data.get("slots", [])
        else:
            print(f"Config file {self.config_path} not found. Using default empty slots.")
            self.config_data = {"camera": {}, "slots": []}
            return []

    def save_config(self, slots):
        self.config_data["slots"] = slots
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=4)
        self.parking_slots = slots

    def detect_vehicles(self, image_path):
        image = cv2.imread(image_path)
        if image is None: return []

        crop_coords = self.config_data.get("crop")
        crop_x, crop_y = 0, 0
        if crop_coords and len(crop_coords) == 4:
            x1, y1, x2, y2 = crop_coords
            image = image[y1:y2, x1:x2]
            crop_x, crop_y = x1, y1

        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        H, W = image.shape[:2]

        TILE_SIZE = 512
        OVERLAP = 0.3
        CONF_THRES = 0.15
        IOU_NMS = 0.45

        stride = int(TILE_SIZE * (1 - OVERLAP))
        tiles = []

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                tile = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
                tiles.append((tile, x, y))

        all_boxes = []
        all_scores = []

        for tile, x_offset, y_offset in tiles:
            results = self.model(tile, conf=CONF_THRES, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]

                if label not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(conf)

        if not all_boxes:
            return []

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)

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

        indices = all_scores.argsort()[::-1]
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            rest = indices[1:]
            ious = compute_iou(all_boxes[current], all_boxes[rest])
            indices = rest[ious < IOU_NMS]

        final_boxes = all_boxes[keep]
        # Scale back to original cropped coordinates
        final_boxes = final_boxes / 1.5
        
        # Add offset back for global image coordinates
        if crop_x > 0 or crop_y > 0:
            final_boxes[:, 0] += crop_x
            final_boxes[:, 1] += crop_y
            final_boxes[:, 2] += crop_x
            final_boxes[:, 3] += crop_y

        return final_boxes

    def extract_slot_roi(self, img, points, erode_iter=3):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        if erode_iter > 0:
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=erode_iter)

        x, y, w, h = cv2.boundingRect(pts)
        roi = img[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        return roi, roi_mask

    def check_slots(self, image_path, output_path="result.jpg"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        slot_status = []
        vehicles = []
        if self.use_yolo:
            vehicles = self.detect_vehicles(image_path)

        for slot in self.parking_slots:
            name = slot.get("name", "Unknown")
            points = np.array(slot["points"], dtype=np.int32)

            is_occupied = False

            if self.use_yolo:
                for v_box in vehicles:
                    v_center = (int((v_box[0] + v_box[2]) / 2), int((v_box[1] + v_box[3]) / 2))
                    if cv2.pointPolygonTest(points, v_center, False) >= 0:
                        is_occupied = True
                        break
            else:
                # Extract ROI
                roi_gray, roi_mask = self.extract_slot_roi(gray, slot["points"])

                # Use Edge Density and Grayscale Standard Deviation as heuristics
                edges = cv2.Canny(roi_gray, 50, 150)
                edges_masked = cv2.bitwise_and(edges, edges, mask=roi_mask)
                area = cv2.countNonZero(roi_mask)

                edge_density = cv2.countNonZero(edges_masked) / area if area > 0 else 0

                mean, stddev = cv2.meanStdDev(roi_gray, mask=roi_mask)
                std_val = stddev[0][0] if stddev is not None else 0

                if edge_density > self.edge_thresh or std_val > self.std_thresh:
                    is_occupied = True

            status = "Occupied" if is_occupied else "Free"
            slot_status.append({"name": name, "status": status})

            # Draw slot box
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            # Make the polygon thicker if occupied
            thickness = 2
            cv2.polylines(img, [points], True, color, thickness)

            # Add small text background for readability
            first_pt = (points[0][0], points[0][1] - 10)
            (w, h), _ = cv2.getTextSize(f"{name}: {status}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (first_pt[0], first_pt[1] - h), (first_pt[0] + w, first_pt[1] + 5), (0, 0, 0), -1)
            cv2.putText(img, f"{name}: {status}", first_pt,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.use_yolo:
            for v_box in vehicles:
                cv2.rectangle(img, (int(v_box[0]), int(v_box[1])), (int(v_box[2]), int(v_box[3])), (255, 0, 0), 2)

        # Ensure directory exists for output
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #cv2.imwrite(output_path, img)
        
        crop_coords = self.config_data.get("crop")
        if crop_coords and len(crop_coords) == 4:
            x1, y1, x2, y2 = crop_coords
            crop_img = img[y1:y2, x1:x2]
            name, ext = os.path.splitext(output_path)
#            cv2.imwrite(f"{name}_crop{ext}", crop_img)
            cv2.imwrite(output_path, crop_img)
            
        return slot_status

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Parking Slot Detector")
    parser.add_argument("--testset", action="store_true", help="Run accuracy evaluation on testset/ directory")
    parser.add_argument("--yolo", action="store_true", help="Use YOLOv8 instead of simple heuristics")
    parser.add_argument("--image", type=str, default="test_capture.jpg", help="Image to process")
    args = parser.parse_args()

    if not os.path.exists("config.json"):
        import sys
        print("Error: 'config.json' does not exist. Please run calibrate.py first.")
        sys.exit(1)

    detector = ParkingDetector(use_yolo=args.yolo)

    if args.testset:
        test_images = glob.glob("testset/*.jpg")
        if not test_images:
            print("No images found in testset/ directory.")
        else:
            correct = 0
            total = 0
            print("Testing detector accuracy on testset...\n")
            for img_path in test_images:
                basename = os.path.basename(img_path)
                parts = basename.replace(".jpg", "").split("_e_")
                true_empty_slots = [int(x) for x in parts[1].split("_")] if len(parts) > 1 else []

                results = detector.check_slots(img_path, output_path=f"result_{basename}")
                if results is None: continue

                for i, res in enumerate(results):
                    slot_idx = i + 1
                    predicted_empty = (res["status"] == "Free")
                    is_true_empty = (slot_idx in true_empty_slots)

                    total += 1
                    if predicted_empty == is_true_empty:
                        correct += 1
                    else:
                        print(f"❌ Error in {basename} Slot {slot_idx}: Predicted Free={predicted_empty}, True Free={is_true_empty}")

            if total > 0:
                print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    else:
        status = detector.check_slots(args.image)
        for s in status:
            print(f"{s['name']}: {s['status']}")
