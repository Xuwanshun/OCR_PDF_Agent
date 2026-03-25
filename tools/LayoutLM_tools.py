from transformers import LayoutLMv3ForTokenClassification
from layoutreader.v3.helpers import prepare_inputs, boxes2inputs, parse_logists
from paddleocr import LayoutDetection

model_slug = "hantian/layoutreader"
layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_slug)
layout_engine = LayoutDetection()

def get_reading_order(ocr_regions):
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1
    image_height = max_y * 1.1

    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)

    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    reading_order = parse_logists(logits, len(boxes))

    return reading_order

def get_ordered_text(ocr_regions, reading_order):
    indexed_regions = [(reading_order[i], i ,ocr_regions[i]) for i in range(len(ocr_regions))]
    indexed_regions.sort(key=lambda x: x[0])
    ordered_text = []
    for position, original_idex, region in indexed_regions:
        ordered_text.append({
            "position": position,
            "text": region.text,
            "confidence": region.confidence,
            "bbox": region.bbox_xyxy,
        })
    
    return ordered_text

def process_document(image_path):
    layout_result = layout_engine.predict(image_path)

    regions = []
    for box in layout_result[0]['boxes']:
        regions.append({
            'lable': box['label'],
            'score': box['score'],
            'bbox': box['coordinate'],
        })
    
    regions = sorted(regions, key=lambda x: x['score', reverse=True])
    return regions
