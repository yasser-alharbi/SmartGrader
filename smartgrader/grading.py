# -*- coding: utf-8 -*-
"""
LLM-based answer evaluation.

An Agno Agent backed by GPT-4 Turbo grades each student answer against
the question using a lenient, encouraging rubric.
"""

import os
import re

from agno.agent import Agent
from agno.models.openai import OpenAIChat

import config


def create_evaluation_agent(max_score=None):
    """Create and return an Agno Agent configured for grading."""
    if max_score is None:
        max_score = config.DEFAULT_MAX_SCORE

    evaluation_agent = Agent(
        model=OpenAIChat(
            id=config.LLM_MODEL_ID,
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        instructions=[
            f"You are a kind and encouraging teacher evaluating student answers.",
            f"You will receive a 'question' and a 'student_answer'.",
            f"The maximum score for this question is {max_score} points.",
            "",
            "IMPORTANT Evaluation Guidelines:",
            "1. Be VERY LENIENT - if the student shows basic understanding, give high marks",
            "2. IGNORE spelling errors completely - focus only on concepts",
            "3. If the core concept is correct, give at least 70% of the score",
            "4. Give full credit if the main idea is present, even if explanation is simple",
            "5. Only deduct points for completely wrong or missing concepts",
            "",
            "Scoring Guidelines:",
            "- 80-100%: Student understands the core concept (even with poor spelling/grammar)",
            "- 60-79%: Student has partial understanding",
            "- 40-59%: Student attempted but missed key concepts",
            "- 0-39%: Answer is completely wrong or unrelated",
            "",
            "Provide your response in this exact format:",
            f"Score: X/{max_score}",
            "Feedback: [One encouraging sentence about what they got right]",
            "Key_Points: [Brief note about important concepts covered or missed]",
            "",
            "Keep feedback SHORT, POSITIVE, and ENCOURAGING.",
        ],
        markdown=True,
    )
    return evaluation_agent


def evaluate_answer(question, student_answer, max_score=None):
    """
    Evaluate a single student answer.

    Args:
        question: The question text.
        student_answer: The student's answer text.
        max_score: Maximum possible points.

    Returns:
        dict with keys: score, max_score, percentage, feedback,
                        key_points, full_evaluation, status
    """
    if max_score is None:
        max_score = config.DEFAULT_MAX_SCORE

    agent = create_evaluation_agent(max_score)
    prompt = f"Question: {question}\nStudent Answer: {student_answer}"

    try:
        result = agent.run(prompt)
        evaluation_text = result.content

        # Parse the structured response
        score_match = re.search(
            r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*\d+", evaluation_text
        )
        score = float(score_match.group(1)) if score_match else 0

        feedback_match = re.search(
            r"Feedback:\s*(.+?)(?=Key_Points:|$)", evaluation_text, re.DOTALL
        )
        feedback = (
            feedback_match.group(1).strip() if feedback_match else "Good effort!"
        )

        key_points_match = re.search(
            r"Key_Points:\s*(.+?)$", evaluation_text, re.DOTALL
        )
        key_points = (
            key_points_match.group(1).strip() if key_points_match else ""
        )

        percentage = (score / max_score) * 100

        return {
            "score": score,
            "max_score": max_score,
            "percentage": percentage,
            "feedback": feedback,
            "key_points": key_points,
            "full_evaluation": evaluation_text,
            "status": (
                "Excellent"
                if percentage >= 90
                else "Good"
                if percentage >= 70
                else "Fair"
                if percentage >= 50
                else "Needs Improvement"
            ),
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "score": 0,
            "max_score": max_score,
            "percentage": 0,
            "feedback": f"Error during evaluation: {e}",
            "key_points": "",
            "full_evaluation": "",
            "status": "Error",
        }
