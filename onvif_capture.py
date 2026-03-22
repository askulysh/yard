import cv2
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera
import os

class ONVIFCapture:
    def __init__(self, ip, user, password, port=80):
        self.ip = ip
        self.user = user
        self.password = password
        self.port = port
        self.camera = None
        self.snapshot_uri = None

    def connect(self):
        try:
            # onvif-zeep requires wsdl files. Default path is typically handled by pkg_resources
            # If not, we might need to specify it manually.
            self.camera = ONVIFCamera(self.ip, self.port, self.user, self.password)
            print(f"Connected to ONVIF camera at {self.ip}")
            return True
        except Exception as e:
            print(f"Error connecting to ONVIF camera: {e}")
            return False

    def get_snapshot_uri(self):
        if not self.camera:
            if not self.connect():
                return None

        try:
            media_service = self.camera.create_media_service()
            profiles = media_service.GetProfiles()
            token = profiles[0].token

            # Get snapshot URI
            res = media_service.GetSnapshotUri({'ProfileToken': token})
            self.snapshot_uri = res.Uri
            print(f"Snapshot URI: {self.snapshot_uri}")
            return self.snapshot_uri
        except Exception as e:
            print(f"Error getting snapshot URI: {e}")
            return None

    def capture_frame(self, output_path="captured_frame.jpg"):
        uri = self.get_snapshot_uri()
        if not uri:
            return False

        try:
            # Some cameras use Digest Auth, others Basic. HTTPDigestAuth is common.
            # We try with the provided credentials.
            # Replace localhost in URI if camera returns it incorrectly
            if "127.0.0.1" in uri:
                uri = uri.replace("127.0.0.1", self.ip)
            elif "localhost" in uri:
                uri = uri.replace("localhost", self.ip)

            response = requests.get(uri, auth=HTTPDigestAuth(self.user, self.password), timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"Frame captured and saved to {output_path}")
                return True
            else:
                # Try Basic Auth if Digest fails
                response = requests.get(uri, auth=(self.user, self.password), timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Frame captured (Basic Auth) and saved to {output_path}")
                    return True
                else:
                    print(f"Failed to capture frame. Status code: {response.status_code}")
                    return False
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return False

if __name__ == "__main__":
    import json

    with open("config.json", "r") as f:
        config = json.load(f)
    camera_cfg = config["camera"]

    # Test connection
    cap = ONVIFCapture(camera_cfg["ip"], camera_cfg["user"], camera_cfg["password"])
    if cap.connect():
        cap.capture_frame("test_capture.jpg")
