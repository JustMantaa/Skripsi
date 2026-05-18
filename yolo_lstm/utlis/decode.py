import base64
import cv2
import numpy as np


def decode_data_url(data_url):
    """Decode a data URL to an OpenCV BGR image."""
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame
