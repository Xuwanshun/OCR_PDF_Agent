# cropping regions for agent tools
import base64
from io import BytesIO

def crop_region(image, bbox, padding = 10):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64decode(buffer.getvalue()).decode('utf-8')

