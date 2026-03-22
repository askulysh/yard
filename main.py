import time
import json
import os
from onvif_capture import ONVIFCapture
from detector import ParkingDetector

def main():
    # Load Configuration
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found.")
        print("Please define your camera credentials and tracking points in config.json.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    camera_cfg = config.get("camera", {})
    try:
        CAMERA_IP = camera_cfg["ip"]
        USER = camera_cfg["user"]
        PASSWORD = camera_cfg["password"]
    except KeyError as e:
        print(f"Error: Missing camera credential {e} in {config_path}")
        return

    CAPTURE_INTERVAL = 30 # seconds

    # Initialize components
    cap = ONVIFCapture(CAMERA_IP, USER, PASSWORD)
    detector = ParkingDetector(model_path="yolov8l-visdrone.pt", use_yolo=True, config_path=config_path)

    print("Starting Parking Monitoring...")
    try:
        while True:
            # 1. Capture Image
            img_name = "latest_capture.jpg"
            if cap.capture_frame(img_name):
                # 2. Detect and Check Slots
                status = detector.check_slots(img_name, output_path="latest_result.jpg")

                # 3. Report Status
                print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                for s in status:
                    print(f"  {s['name']}: {s['status']}")

                # Save status to JSON for the Telegram Bot
                with open("status.json", "w") as f:
                    json.dump(status, f, indent=4)

                print(f"Result saved to latest_result.jpg and status.json")
            else:
                print("Failed to capture image. Retrying in next cycle.")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    main()
