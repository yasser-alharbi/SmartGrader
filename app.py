# -*- coding: utf-8 -*-
"""
SmartGrader – Gradio Application

Launch with:
    python app.py

Upload an exam paper image → adjust confidence and max score →
click **Grade** to see AI-generated scores, feedback, and a full report.
"""

import os
import tempfile
import warnings

import cv2
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import torch

import config
from smartgrader.detection import load_roboflow_model
from smartgrader.ocr import init_ocr_models
from smartgrader.pipeline import grade_qa_with_matching
from smartgrader.visualization import (
    create_comprehensive_visualization,
    create_detailed_report,
)

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ── Lazy globals (loaded once on first use) ──────────────────────────────────
_model = None
_ocr_models = None


def _ensure_models():
    """Load Roboflow + OCR models on first call, then reuse."""
    global _model, _ocr_models
    if _model is None:
        print("Loading Roboflow model...")
        _model = load_roboflow_model()
    if _ocr_models is None:
        _ocr_models = init_ocr_models()
    return _model, _ocr_models


# ═══════════════════════════════════════════════════════════════════════════
# Gradio callback helpers
# ═══════════════════════════════════════════════════════════════════════════


def test_model_detection(image, confidence=47):
    """
    Run Roboflow detection only (no OCR / grading) and return an
    annotated figure + info text.
    """
    model, _ = _ensure_models()

    # Write the uploaded image to a temp file
    temp_path = os.path.join(tempfile.gettempdir(), "sg_test_detection.jpg")
    if hasattr(image, "save"):
        image.save(temp_path)
    else:
        cv2.imwrite(temp_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    try:
        prediction = model.predict(
            temp_path, confidence=int(confidence), overlap=config.DEFAULT_OVERLAP
        )
        detections = prediction.json()["predictions"]

        fig, ax = plt.subplots(figsize=(14, 10))
        img = cv2.imread(temp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        colors = {
            "Printed": "#1976D2",
            "Handwritten": "#D32F2F",
            "MachineText": "#1976D2",
            "HandWritten": "#D32F2F",
        }

        for det in detections:
            x_center, y_center = det["x"], det["y"]
            w, h = det["width"], det["height"]
            x1, y1 = x_center - w / 2, y_center - h / 2
            color = colors.get(det["class"], "#388E3C")

            rect = plt.Rectangle(
                (x1, y1), w, h,
                linewidth=2, edgecolor=color, facecolor="none", alpha=0.8,
            )
            ax.add_patch(rect)

            label = f"{det['class']} {det['confidence']:.1f}%"
            ax.text(
                x1, y1 - 5, label, color=color, fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            )

        ax.set_title(
            f"Detection Test – Found {len(detections)} regions "
            f"(Confidence: {int(confidence)}%)",
            fontsize=14, weight="bold",
        )
        ax.axis("off")

        # Build info string
        class_counts = {}
        for det in detections:
            class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1

        info = (
            f"📊 Detection Test Results:\n{'=' * 40}\n"
            f"Total Detections: {len(detections)}\n"
            f"Confidence Threshold: {int(confidence)}%\n\n"
            f"Breakdown by Class:\n{'=' * 40}"
        )
        for cname, count in class_counts.items():
            icon = "📘" if cname in ("Printed", "MachineText") else "✍️"
            info += f"\n{icon} {cname}: {count} regions"

        info += (
            "\n\n✅ API Status: Working"
            f"\n📡 Model: Version {config.ROBOFLOW_MODEL_VERSION}"
            "\n🎯 mAP@50: 63.1%"
        )

    except Exception as e:
        fig, ax = plt.subplots(figsize=(12, 8))
        if os.path.exists(temp_path):
            img = cv2.imread(temp_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(f"Detection Error: {e}", color="red")
        ax.axis("off")

        info = (
            f"❌ Detection Test Failed:\n{'=' * 40}\n"
            f"Error: {e}\n\n"
            "Troubleshooting:\n"
            "1. Check Roboflow API connection\n"
            "2. Verify API key is valid\n"
            "3. Check image format"
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return fig, info


def gradio_grade_interface(image, max_score, confidence):
    """
    Gradio callback for the **🎯 Grade Paper** button.

    Returns:
        (figure, summary_text, detailed_report)
    """
    if image is None:
        return None, "❌ Please upload a Q&A paper image.", ""

    model, ocr_models = _ensure_models()

    # Save uploaded image to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "sg_qa_input.jpg")
    try:
        if hasattr(image, "save"):
            image.save(temp_path)
        else:
            cv2.imwrite(
                temp_path,
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
            )
    except Exception as e:
        return None, f"❌ Failed to process image: {e}", ""

    # Run grading pipeline
    try:
        results = grade_qa_with_matching(
            temp_path,
            model,
            ocr_models,
            confidence=int(confidence),
            max_score_per_question=int(max_score),
        )
    except Exception as e:
        return None, f"❌ Grading pipeline error: {e}", ""

    if isinstance(results, dict) and results.get("error"):
        return None, f"❌ {results['error']}", ""

    # Build outputs
    try:
        fig = create_comprehensive_visualization(results, temp_path)
        summary = (
            f"Grade: {results['grade']} "
            f"({results['overall_percentage']:.1f}%)\n"
            f"Total: {results['total_score']:.1f}"
            f"/{results['total_possible']}\n"
            f"Questions graded: {results['num_questions']}"
        )
        detailed = create_detailed_report(results)
    except Exception as e:
        return None, f"❌ Visualization/report error: {e}", ""

    return fig, summary, detailed


# ═══════════════════════════════════════════════════════════════════════════
# Gradio Interface
# ═══════════════════════════════════════════════════════════════════════════

def build_app():
    """Construct and return the Gradio Blocks app."""

    with gr.Blocks(
        title="📚 English Q&A Auto-Grader",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # 📚 SmartGrader: an English Q&A Auto-Grader with AI Evaluation
            ### Powered by YOLOv11, OCR, and GPT-4

            ---

            ## 🚀 Features:
            - **🔍 Smart Detection**: Uses Roboflow YOLOv11 to detect printed questions and handwritten answers
            - **🧩 Intelligent Matching**: AI matches questions with their corresponding answers
            - **📝 Dual OCR**: EasyOCR for printed text, TrOCR for handwritten text
            - **🤖 AI Grading**: GPT-4 evaluates answers and provides detailed feedback
            - **📊 Comprehensive Reports**: Detailed visualizations and feedback for each question

            ---
            """
        )

        # ── Tab 1: Grade Paper ────────────────────────────────────────────
        with gr.Tab("🎯 Grade Paper"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload & Configure")

                    input_image = gr.Image(
                        type="pil",
                        label="Upload Q&A Paper",
                        height=400,
                    )

                    with gr.Row():
                        max_score_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config.DEFAULT_MAX_SCORE,
                            step=1,
                            label="Max Score per Question",
                            info="Points for each question",
                        )
                        confidence_slider = gr.Slider(
                            minimum=10,
                            maximum=90,
                            value=config.DEFAULT_CONFIDENCE,
                            step=1,
                            label="Detection Confidence %",
                            info="Lower = more detections, higher = stricter",
                        )

                    grade_btn = gr.Button(
                        "🎯 Grade Paper",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    output_plot = gr.Plot(label="Grading Visualization")
                    output_summary = gr.Textbox(
                        label="Grade Summary",
                        lines=4,
                        interactive=False,
                    )
                    output_report = gr.Textbox(
                        label="Detailed Report (copy-friendly)",
                        lines=15,
                        interactive=False,
                        show_copy_button=True,
                    )

            grade_btn.click(
                fn=gradio_grade_interface,
                inputs=[input_image, max_score_slider, confidence_slider],
                outputs=[output_plot, output_summary, output_report],
            )

        # ── Tab 2: Test Detection ─────────────────────────────────────────
        with gr.Tab("🔬 Test Detection"):
            gr.Markdown(
                "### Test the detection model without running the full grading pipeline"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    test_image = gr.Image(
                        type="pil",
                        label="Upload Test Image",
                        height=400,
                    )
                    test_confidence = gr.Slider(
                        minimum=10,
                        maximum=90,
                        value=config.DEFAULT_CONFIDENCE,
                        step=1,
                        label="Detection Confidence %",
                    )
                    test_btn = gr.Button(
                        "🔍 Run Detection",
                        variant="primary",
                    )

                with gr.Column(scale=2):
                    test_plot = gr.Plot(label="Detection Results")
                    test_info = gr.Textbox(
                        label="Detection Info",
                        lines=12,
                        interactive=False,
                    )

            test_btn.click(
                fn=test_model_detection,
                inputs=[test_image, test_confidence],
                outputs=[test_plot, test_info],
            )

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            ### 👥 Project Members
            **Yasir Alharbi** · **Naif Alharbi** · **Sultan Alaqili**

            *Tech stack: YOLOv11 (Roboflow) · EasyOCR · TrOCR · Agno + GPT-4 · OpenCV · Matplotlib · Gradio*
            """
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True)
