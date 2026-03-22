import cv2
import json
import os
import numpy as np
from onvif_capture import ONVIFCapture

def split_area(p1, p2, p3, p4, n):
    slots = []
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n

        tp0 = [p1[0] + t0 * (p4[0] - p1[0]), p1[1] + t0 * (p4[1] - p1[1])]
        tp1 = [p1[0] + t1 * (p4[0] - p1[0]), p1[1] + t1 * (p4[1] - p1[1])]

        bp0 = [p2[0] + t0 * (p3[0] - p2[0]), p2[1] + t0 * (p3[1] - p2[1])]
        bp1 = [p2[0] + t1 * (p3[0] - p2[0]), p2[1] + t1 * (p3[1] - p2[1])]

        slot_points = [
            [int(round(tp0[0])), int(round(tp0[1]))],
            [int(round(tp1[0])), int(round(tp1[1]))],
            [int(round(bp1[0])), int(round(bp1[1]))],
            [int(round(bp0[0])), int(round(bp0[1]))]
        ]

        slots.append({
            "name": f"Slot {i+1}",
            "points": slot_points
        })
    return slots

def calibrate(area_points=None, num_slots=None):
    # Load camera credentials from config.json
    camera_cfg = {}
    if os.path.exists(DEFAULT_JSON):
        with open(DEFAULT_JSON, "r") as f:
            config_data = json.load(f)
            if isinstance(config_data, dict):
                camera_cfg = config_data.get("camera", {})

    try:
        CAMERA_IP = camera_cfg["ip"]
        USER = camera_cfg["user"]
        PASSWORD = camera_cfg["password"]
    except KeyError as e:
        import sys
        print(f"Error: Missing camera credential {e} in {DEFAULT_JSON}")
        sys.exit(1)

    # If parameters provided, update config.json
    if area_points and num_slots:
        print(f"Splitting area into {num_slots} slots...")
        p1, p2, p3, p4 = area_points
        new_slots_list = split_area(p1, p2, p3, p4, num_slots)

        config_data = {}
        if os.path.exists(DEFAULT_JSON):
            with open(DEFAULT_JSON, "r") as f:
                config_data = json.load(f)

        config_data["slots"] = new_slots_list

        with open(DEFAULT_JSON, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"Updated '{DEFAULT_JSON}' with new split slots.")

    cap = ONVIFCapture(CAMERA_IP, USER, PASSWORD)
    img_name = "calibration.jpg"

    print(f"Capturing calibration image from {CAMERA_IP}...")
    if cap.capture_frame(img_name):
        print(f"Success! Image saved as '{img_name}'. Drawing current slots...")

        # Draw current slots
        img = cv2.imread(img_name)
        if os.path.exists("parking_slots.json"):
            with open("parking_slots.json", "r") as f:
                slots = json.load(f)

            for slot in slots:
                pts = np.array(slot["points"], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)

                # Add label
                if slot["points"]:
                    first_pt = slot["points"][0]
                    cv2.putText(img, slot["name"], (first_pt[0], first_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imwrite("ttt.jpg", img)
            print(f"Updated '{img_name}' (as ttt.jpg) with {len(slots)} slots.")

        print("Find the corners (x, y) for each parking slot.")
        print("Update 'parking_slots.json' with these coordinates.")
        print("\nExample entry for a slot:")
        print(" {")
        print("  \"name\": \"My Slot\",")
        print("  \"points\": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")
        print(" }")
    else:
        print("Failed to capture calibration image. Check your camera connection and credentials.")

if __name__ == "__main__":
    # Example: calibrate([[562, 122], [561, 160], [786, 173], [787, 129]], 12)
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate parking slots.")
    parser.add_argument("--split", action="store_true", help="Split an area into slots")
    parser.add_argument("--points", type=str, help="Area points in format '[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]'")
    parser.add_argument("--num", type=int, default=12, help="Number of slots to split into")

    args = parser.parse_args()

    if args.split and args.points:
        import ast
        points = ast.literal_eval(args.points)
        calibrate(area_points=points, num_slots=args.num)
    else:
        calibrate()
