import glob
import os
from detector import ParkingDetector

detector = ParkingDetector(config_path="parking_slots.json", edge_thresh=0.15, std_thresh=25.0)
test_images = glob.glob("testset/*.jpg")

correct = 0
total = 0

print("Testing new detector...\n")
for img_path in test_images:
    basename = os.path.basename(img_path)
    # Parse true empty slots
    parts = basename.replace(".jpg", "").split("_e_")
    true_empty_slots = []
    if len(parts) > 1:
        true_empty_slots = [int(x) for x in parts[1].split("_")]

    # Run detector
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

print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")
