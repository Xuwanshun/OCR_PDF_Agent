from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

@dataclass
class OCRRegion:
    text: str
    bbox: list
    confidence: float

    @property
    def bbox_xyxy(self):
        """Returen bbox as [x1, y1, x2, y2] format"""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
