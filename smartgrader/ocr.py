# -*- coding: utf-8 -*-
"""
OCR text extraction.

* **EasyOCR** – used for *Printed* text regions.
* **TrOCR** (Transformer-based, Microsoft) – used for *Handwritten* text.
"""

import cv2
import torch
import easyocr
from PIL import Image as PILImage
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import config


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def init_ocr_models():
    """
    Load EasyOCR reader and TrOCR model/processor.

    Returns:
        dict with keys: easy_reader, trocr_processor, trocr_model
    """
    print("Loading EasyOCR for printed text...")
    easy_reader = easyocr.Reader(
        config.EASYOCR_LANGUAGES,
        gpu=torch.cuda.is_available(),
    )

    print("Loading TrOCR for handwritten text...")
    processor = TrOCRProcessor.from_pretrained(config.TROCR_MODEL_NAME)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(config.TROCR_MODEL_NAME)

    if torch.cuda.is_available():
        trocr_model = trocr_model.cuda()

    print("Both OCR models loaded successfully!")

    return {
        "easy_reader": easy_reader,
        "trocr_processor": processor,
        "trocr_model": trocr_model,
    }


# ---------------------------------------------------------------------------
# Per-region extraction
# ---------------------------------------------------------------------------

def extract_printed_text(image, bbox, easy_reader, padding=5):
    """
    Extract printed text from a bounding-box region using EasyOCR.

    Args:
        image: RGB numpy array.
        bbox: [x1, y1, x2, y2].
        easy_reader: An initialised EasyOCR Reader.
        padding: Extra pixels around the crop.

    Returns:
        str: Recognised text.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    word_region = image[y1:y2, x1:x2]

    # Convert to grayscale and enhance
    if len(word_region.shape) == 3:
        gray = cv2.cvtColor(word_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = word_region

    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)

    try:
        results = easy_reader.readtext(enhanced, detail=0)
        text = results[0] if results else ""
    except Exception:
        text = ""

    return text.strip()


def extract_handwritten_text(image, bbox, trocr_processor, trocr_model, padding=10):
    """
    Extract handwritten text from a bounding-box region using TrOCR.

    Args:
        image: RGB numpy array.
        bbox: [x1, y1, x2, y2].
        trocr_processor: TrOCR processor.
        trocr_model: TrOCR model.
        padding: Extra pixels around the crop.

    Returns:
        str: Recognised text.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    word_region = image[y1:y2, x1:x2]
    pil_image = PILImage.fromarray(word_region)

    pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()

    try:
        generated_ids = trocr_model.generate(pixel_values)
        generated_text = trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        text = generated_text.strip()
    except Exception as e:
        print(f"TrOCR error: {e}")
        text = ""

    return text


def extract_word_text(image, bbox, is_handwritten, ocr_models):
    """
    Dispatch to the appropriate OCR engine.

    Args:
        image: RGB numpy array.
        bbox: [x1, y1, x2, y2].
        is_handwritten: True → TrOCR, False → EasyOCR.
        ocr_models: dict returned by `init_ocr_models`.

    Returns:
        str: Recognised text.
    """
    if is_handwritten:
        return extract_handwritten_text(
            image,
            bbox,
            ocr_models["trocr_processor"],
            ocr_models["trocr_model"],
            padding=10,
        )
    else:
        return extract_printed_text(
            image,
            bbox,
            ocr_models["easy_reader"],
            padding=5,
        )
