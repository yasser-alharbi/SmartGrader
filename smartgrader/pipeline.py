# -*- coding: utf-8 -*-
"""
End-to-end grading pipeline.

Orchestrates: Detection → OCR → Sorting → Matching → Evaluation.
"""

import cv2

import config
from smartgrader.ocr import extract_word_text
from smartgrader.sorting import group_regions_by_lines
from smartgrader.matching import match_qa_pairs
from smartgrader.grading import evaluate_answer


def process_multiple_qa(image_path, model, ocr_models, confidence=None):
    """
    Detect, OCR and group all questions and answers in an image.

    Args:
        image_path: Path to the exam paper image.
        model: Roboflow model object.
        ocr_models: dict from ``ocr.init_ocr_models()``.
        confidence: Detection confidence threshold.

    Returns:
        dict with keys: questions, answers, image (RGB numpy), prediction.
    """
    if confidence is None:
        confidence = config.DEFAULT_CONFIDENCE

    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with Roboflow
    print(f"Detecting words with confidence threshold: {confidence}%")
    prediction = model.predict(
        image_path,
        confidence=confidence,
        overlap=config.DEFAULT_OVERLAP,
    )

    printed_regions = []
    handwritten_regions = []

    for pred in prediction.json()["predictions"]:
        x_center = pred["x"]
        y_center = pred["y"]
        w = pred["width"]
        h = pred["height"]

        bbox = [
            x_center - w / 2,
            y_center - h / 2,
            x_center + w / 2,
            y_center + h / 2,
        ]

        is_handwritten = pred["class"] in ["Handwritten"]
        word_text = extract_word_text(
            image_rgb, bbox, is_handwritten=is_handwritten, ocr_models=ocr_models
        )

        region_info = {
            "text": word_text,
            "bbox": bbox,
            "center_x": x_center,
            "center_y": y_center,
            "confidence": pred["confidence"],
        }

        if pred["class"] in ["Printed"]:
            printed_regions.append(region_info)
        else:
            handwritten_regions.append(region_info)

    questions = group_regions_by_lines(printed_regions)
    answers = group_regions_by_lines(handwritten_regions)

    return {
        "questions": questions,
        "answers": answers,
        "image": image_rgb,
        "prediction": prediction,
    }


def grade_qa_with_matching(
    image_path,
    model,
    ocr_models,
    confidence=None,
    max_score_per_question=None,
):
    """
    Full grading pipeline: detect → OCR → match → evaluate.

    Args:
        image_path: Path to the exam paper image.
        model: Roboflow model object.
        ocr_models: dict from ``ocr.init_ocr_models()``.
        confidence: Detection confidence threshold.
        max_score_per_question: Maximum points per question.

    Returns:
        dict with keys: evaluations, total_score, total_possible,
                        overall_percentage, grade, num_questions,
                        extraction_result.
    """
    if confidence is None:
        confidence = config.DEFAULT_CONFIDENCE
    if max_score_per_question is None:
        max_score_per_question = config.DEFAULT_MAX_SCORE

    # Step 1 — Extract questions and answers
    print("📝 Step 1: Extracting questions and answers from image...")
    extraction_result = process_multiple_qa(
        image_path, model, ocr_models, confidence
    )

    questions = extraction_result["questions"]
    answers = extraction_result["answers"]

    print(
        f"Found {len(questions)} question regions and "
        f"{len(answers)} answer regions"
    )

    if not questions:
        return {
            "error": "No questions found in image",
            "evaluations": [],
            "total_score": 0,
            "total_possible": 0,
            "overall_percentage": 0,
            "grade": "N/A",
        }

    # Step 2 — Match Q&A
    print("\n🔄 Step 2: Matching questions with answers using AI...")
    qa_pairs = match_qa_pairs(questions, answers)
    print(f"Successfully matched {len(qa_pairs)} Q&A pairs")

    # Step 3 — Evaluate
    print("\n🤖 Step 3: Evaluating each answer...")
    evaluations = []
    total_score = 0

    for qa in qa_pairs:
        print(f"\nEvaluating Question {qa['question_number']}...")
        print(f"Q: {qa['question'][:50]}...")
        print(f"A: {qa['answer'][:50]}...")

        eval_result = evaluate_answer(
            qa["question"], qa["answer"], max_score_per_question
        )

        eval_result["question"] = qa["question"]
        eval_result["student_answer"] = qa["answer"]
        eval_result["question_number"] = qa["question_number"]

        evaluations.append(eval_result)
        total_score += eval_result["score"]

        print(f"Score: {eval_result['score']}/{max_score_per_question}")

    # Overall grade
    total_possible = len(qa_pairs) * max_score_per_question
    overall_percentage = (
        (total_score / total_possible * 100) if total_possible > 0 else 0
    )

    if overall_percentage >= 90:
        grade = "A"
    elif overall_percentage >= 80:
        grade = "B"
    elif overall_percentage >= 70:
        grade = "C"
    elif overall_percentage >= 60:
        grade = "D"
    else:
        grade = "F"

    print(f"\n✅ Grading Complete!")
    print(f"Overall Grade: {grade} ({overall_percentage:.1f}%)")

    return {
        "evaluations": evaluations,
        "total_score": total_score,
        "total_possible": total_possible,
        "overall_percentage": overall_percentage,
        "grade": grade,
        "num_questions": len(qa_pairs),
        "extraction_result": extraction_result,
    }
