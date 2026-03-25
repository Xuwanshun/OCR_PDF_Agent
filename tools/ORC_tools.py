from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR
from matplotlib import colormaps
from lanchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from lanchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
import csv
import os
from dotenv import load_dotenv
from langchain.tools import tool
import base64
from lanchain_openai import ChatOpenAI
from typing import List, Dict, Any

_ = load_dotenv(override=True)

ocr = PaddleOCR(lang='en')

@tool
def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    try:
        result = ocr.predict(image_path)
        page = result[0]

        texts = page['rec_texts']
        boxes = page['dt_polys']
        scores = page.get('rec_scores', [None] * len(texts))

        extracted_items = []
        for text, box, score in zip(texts, boxes, scores):
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            item = {
                'text': text,
                'bbox': bbox,
            }
        
        return item
