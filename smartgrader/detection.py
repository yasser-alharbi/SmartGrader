# -*- coding: utf-8 -*-
"""
Roboflow model loading and region detection.

Uses a pre-trained YOLOv11 model hosted on Roboflow to detect
*Printed* and *Handwritten* word-level bounding boxes.
"""

from roboflow import Roboflow
import config


def load_roboflow_model():
    """
    Initialise the Roboflow client and return the prediction model.

    Returns:
        model: A Roboflow model object ready for `.predict()`.
    """
    rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
    workspace = rf.workspace(config.ROBOFLOW_WORKSPACE)
    project = workspace.project(config.ROBOFLOW_PROJECT)
    model = project.version(config.ROBOFLOW_MODEL_VERSION).model
    return model


def detect_regions(image_path, model, confidence=None):
    """
    Run object-detection on an image and return structured detections.

    Args:
        image_path: Path to the image file.
        model: Roboflow model object (from `load_roboflow_model`).
        confidence: Detection confidence threshold (0-100).

    Returns:
        list[dict]: Each dict contains:
            class, confidence, bbox [x1,y1,x2,y2],
            center_x, center_y, width, height
    """
    if confidence is None:
        confidence = config.DEFAULT_CONFIDENCE

    prediction = model.predict(
        image_path,
        confidence=confidence,
        overlap=config.DEFAULT_OVERLAP,
    )

    detections = []
    for pred in prediction.json()["predictions"]:
        x_center = pred["x"]
        y_center = pred["y"]
        w = pred["width"]
        h = pred["height"]

        detections.append(
            {
                "class": pred["class"],
                "confidence": pred["confidence"],
                "bbox": [
                    x_center - w / 2,  # x1
                    y_center - h / 2,  # y1
                    x_center + w / 2,  # x2
                    y_center + h / 2,  # y2
                ],
                "center_x": x_center,
                "center_y": y_center,
                "width": w,
                "height": h,
            }
        )

    return detections
