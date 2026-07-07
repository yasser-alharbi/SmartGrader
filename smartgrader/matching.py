# -*- coding: utf-8 -*-
"""
LLM-based question-answer matching.

An Agno Agent backed by GPT-4 Turbo pairs each detected printed question
with the most likely handwritten answer, using both content relevance and
spatial (vertical) proximity.
"""

import json
import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat

import config


def create_qa_matching_agent():
    """Create and return an Agno Agent configured for Q&A matching."""
    matching_agent = Agent(
        model=OpenAIChat(
            id=config.LLM_MODEL_ID,
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        instructions=[
            "You are an expert at matching questions with their corresponding answers.",
            "You will receive:",
            "1. A list of detected questions (printed text)",
            "2. A list of detected answers (handwritten text)",
            "",
            "Your task is to:",
            "1. Identify how many distinct questions there are",
            "2. Match each question with its most likely answer based on:",
            "   - Content relevance",
            "   - Spatial proximity (answers usually appear below questions)",
            "   - Logical connection",
            "",
            "Return ONLY a JSON response in this exact format:",
            "{",
            '  "qa_pairs": [',
            "    {",
            '      "question_number": 1,',
            '      "question": "complete question text",',
            '      "answer": "complete answer text"',
            "    }",
            "  ]",
            "}",
            "",
            "Important:",
            "- Combine word fragments into complete sentences",
            "- Fix obvious OCR errors if possible",
            "- If multiple answers seem to belong to one question, combine them",
            "- Number questions in order they appear",
        ],
        markdown=False,
    )
    return matching_agent


def match_qa_pairs(questions, answers):
    """
    Use an LLM to match detected questions with their answers.

    Args:
        questions: list[dict] with keys ``text``, ``y_position``, ``bbox``.
        answers:   list[dict] with same keys.

    Returns:
        list[dict]: Each dict has ``question_number``, ``question``, ``answer``.
    """
    if not questions or not answers:
        return []

    agent = create_qa_matching_agent()

    questions_text = "\n".join(
        [
            f"Q{i+1} (y={q['y_position']:.0f}): {q['text']}"
            for i, q in enumerate(questions)
        ]
    )
    answers_text = "\n".join(
        [
            f"A{i+1} (y={a['y_position']:.0f}): {a['text']}"
            for i, a in enumerate(answers)
        ]
    )

    prompt = (
        "Questions detected (with vertical position):\n"
        f"{questions_text}\n\n"
        "Answers detected (with vertical position):\n"
        f"{answers_text}\n\n"
        "Match each question with its corresponding answer based on content and position."
    )

    try:
        result = agent.run(prompt)
        response_text = result.content

        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            matched_pairs = json.loads(json_str)["qa_pairs"]
        else:
            matched_pairs = _fallback_match(questions, answers)

        return matched_pairs

    except Exception as e:
        print(f"Matching error: {e}")
        return _fallback_match(questions, answers)


def _fallback_match(questions, answers):
    """Simple positional fallback when LLM matching fails."""
    pairs = []
    for i, q in enumerate(questions):
        if i < len(answers):
            pairs.append(
                {
                    "question_number": i + 1,
                    "question": q["text"],
                    "answer": answers[i]["text"],
                }
            )
    return pairs
