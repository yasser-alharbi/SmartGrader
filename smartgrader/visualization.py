# -*- coding: utf-8 -*-
"""
Visualisation and reporting utilities.

* ``create_comprehensive_visualization`` – Matplotlib figure with grade
  summary, score bars, and detailed per-question feedback.
* ``create_detailed_report`` – Plain-text grading report.
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Matplotlib figure
# ---------------------------------------------------------------------------

def create_comprehensive_visualization(grading_results, image_path):
    """
    Build a rich Matplotlib figure summarising the grading results.

    Args:
        grading_results: dict returned by ``pipeline.grade_qa_with_matching``.
        image_path: Path to the original exam image.

    Returns:
        matplotlib.figure.Figure
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    num_questions = grading_results["num_questions"]
    fig_height = max(12, 8 + (num_questions * 2))
    fig = plt.figure(figsize=(20, fig_height))

    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1, 1.5], width_ratios=[1, 1, 1.2]
    )

    # ── Top-left: original image ──────────────────────────────────────────
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title("Original Q&A Paper", fontsize=14, weight="bold")
    ax1.axis("off")

    # ── Top-middle: grade circle ──────────────────────────────────────────
    ax2 = plt.subplot(gs[0, 1])
    ax2.axis("off")

    grade_color = {
        "A": "#2E7D32",
        "B": "#558B2F",
        "C": "#F57C00",
        "D": "#E65100",
        "F": "#B71C1C",
        "N/A": "#757575",
    }

    circle = plt.Circle(
        (0.5, 0.7),
        0.25,
        color=grade_color.get(grading_results["grade"], "#757575"),
        alpha=0.2,
    )
    ax2.add_patch(circle)

    ax2.text(
        0.5, 0.7, grading_results["grade"],
        fontsize=48, weight="bold", ha="center", va="center",
        color=grade_color.get(grading_results["grade"], "black"),
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5, 0.4, f"{grading_results['overall_percentage']:.1f}%",
        fontsize=24, ha="center", transform=ax2.transAxes,
    )
    ax2.text(
        0.5, 0.25,
        f"Total: {grading_results['total_score']:.1f}/{grading_results['total_possible']}",
        fontsize=16, ha="center", transform=ax2.transAxes,
    )
    ax2.text(
        0.5, 0.1, f"{num_questions} Questions Graded",
        fontsize=12, ha="center", style="italic", transform=ax2.transAxes,
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # ── Top-right: score bars ─────────────────────────────────────────────
    ax3 = plt.subplot(gs[0, 2])
    ax3.axis("off")
    ax3.text(
        0.5, 0.95, "SCORE BREAKDOWN",
        fontsize=14, weight="bold", ha="center", transform=ax3.transAxes,
    )

    y_pos = 0.85
    for ev in grading_results["evaluations"]:
        if ev["percentage"] >= 80:
            bar_color = "#4CAF50"
        elif ev["percentage"] >= 60:
            bar_color = "#FFC107"
        else:
            bar_color = "#F44336"

        bar_width = ev["percentage"] / 100 * 0.7
        rect = plt.Rectangle(
            (0.1, y_pos - 0.02), bar_width, 0.04,
            facecolor=bar_color, alpha=0.6,
        )
        ax3.add_patch(rect)

        ax3.text(
            0.05, y_pos, f"Q{ev['question_number']}:",
            fontsize=10, weight="bold", transform=ax3.transAxes,
        )
        ax3.text(
            0.85, y_pos, f"{ev['score']:.1f}/{ev['max_score']}",
            fontsize=10, transform=ax3.transAxes,
        )
        y_pos -= 0.08

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # ── Bottom row: detailed feedback ─────────────────────────────────────
    ax4 = plt.subplot(gs[1, :])
    ax4.axis("off")

    ax4.text(
        0.5, 0.98, "📝 DETAILED FEEDBACK FOR EACH QUESTION",
        fontsize=16, weight="bold", ha="center", transform=ax4.transAxes,
    )

    feedback_start_y = 0.92
    space_per_qa = min(0.85 / num_questions, 0.25)

    for i, ev in enumerate(grading_results["evaluations"]):
        y_base = feedback_start_y - (i * space_per_qa)

        score_color = (
            "#4CAF50" if ev["percentage"] >= 70
            else "#FFC107" if ev["percentage"] >= 50
            else "#F44336"
        )

        badge = plt.Rectangle(
            (0.02, y_base - 0.015), 0.08, 0.03,
            facecolor=score_color, alpha=0.3,
        )
        ax4.add_patch(badge)

        ax4.text(
            0.06, y_base, f"Q{ev['question_number']}",
            fontsize=11, weight="bold", ha="center", transform=ax4.transAxes,
        )
        ax4.text(
            0.12, y_base, f"({ev['score']:.1f}/{ev['max_score']})",
            fontsize=10, transform=ax4.transAxes,
        )

        # Question
        q_text = ev["question"]
        if len(q_text) > 80:
            q_text = q_text[:77] + "..."
        ax4.text(
            0.02, y_base - 0.025, f"Question: {q_text}",
            fontsize=9, color="darkblue", transform=ax4.transAxes,
        )

        # Answer
        a_text = ev["student_answer"]
        if len(a_text) > 80:
            a_text = a_text[:77] + "..."
        ax4.text(
            0.02, y_base - 0.045, f"Answer: {a_text}",
            fontsize=9, color="darkred", transform=ax4.transAxes,
        )

        # Feedback
        fb_text = ev["feedback"]
        if len(fb_text) > 100:
            fb_text = fb_text[:97] + "..."
        ax4.text(
            0.02, y_base - 0.065, f"Feedback: {fb_text}",
            fontsize=9, color="green", weight="bold", transform=ax4.transAxes,
        )

        # Key points
        if ev.get("key_points"):
            kp = ev["key_points"]
            if len(kp) > 90:
                kp = kp[:87] + "..."
            ax4.text(
                0.02, y_base - 0.085, f"Key Points: {kp}",
                fontsize=8, color="gray", style="italic",
                transform=ax4.transAxes,
            )

        # Separator
        if i < num_questions - 1:
            ax4.plot(
                [0.02, 0.98], [y_base - 0.095, y_base - 0.095],
                color="lightgray", linewidth=0.5, transform=ax4.transAxes,
            )

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def create_detailed_report(grading_results):
    """
    Generate a plain-text grading report.

    Args:
        grading_results: dict from ``pipeline.grade_qa_with_matching``.

    Returns:
        str: Formatted report text.
    """
    sep = "=" * 70
    report = f"{sep}\n📊 COMPREHENSIVE GRADING REPORT\n{sep}\n\n"

    report += f"Overall Grade: {grading_results['grade']}\n"
    report += (
        f"Total Score: {grading_results['total_score']:.1f}"
        f"/{grading_results['total_possible']}\n"
    )
    report += f"Percentage: {grading_results['overall_percentage']:.1f}%\n"
    report += f"Number of Questions: {grading_results['num_questions']}\n"
    report += f"\n{'-' * 70}\n\n"

    report += f"DETAILED ANALYSIS BY QUESTION:\n{sep}\n\n"

    for ev in grading_results["evaluations"]:
        report += f"QUESTION {ev['question_number']}:\n"
        report += f"{'-' * 40}\n"
        report += f"Question: {ev['question']}\n\n"
        report += f"Student Answer: {ev['student_answer']}\n\n"
        report += (
            f"Score: {ev['score']}/{ev['max_score']} "
            f"({ev['percentage']:.1f}%)\n"
        )
        report += f"Status: {ev['status']}\n\n"
        report += f"Feedback: {ev['feedback']}\n"
        if ev.get("key_points"):
            report += f"\nKey Points: {ev['key_points']}\n"
        report += f"\n{sep}\n\n"

    # Summary statistics
    report += f"SUMMARY STATISTICS:\n{'-' * 40}\n"
    scores = [e["percentage"] for e in grading_results["evaluations"]]
    if scores:
        report += f"Average Score: {sum(scores) / len(scores):.1f}%\n"
        report += f"Highest Score: {max(scores):.1f}%\n"
        report += f"Lowest Score: {min(scores):.1f}%\n"

        for label in ("Excellent", "Good", "Fair", "Needs Improvement"):
            count = sum(
                1 for e in grading_results["evaluations"]
                if e["status"] == label
            )
            if count > 0:
                report += f"{label} Answers: {count}\n"

    report += f"\n{sep}\nEND OF REPORT\n{sep}"
    return report
