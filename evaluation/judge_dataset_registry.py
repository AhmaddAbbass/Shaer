#!/usr/bin/env python3
from __future__ import annotations

DATASET_SPECS = {
    "shaer": {
        "dataset_id": "Shaer-AI/shaer-sft-test",
        "split": "test",
        "poem_field": "generated_text",
        "description_field": "enhanced_description",
        "id_field_candidates": ["id"],
        "judge_metrics": [
            "description_adherence",
            "meaning",
            "fluency",
            "coherence",
            "poeticness",
        ],
    },
    "ashaar": {
        "dataset_id": "Shaer-AI/shaer-eval-ashaar-native-controls",
        "split": "test",
        "poem_field": "generated_text",
        "description_field": "enhanced_description",
        "id_field_candidates": ["generation_id", "id", "input_row_id"],
        "judge_metrics": [
            "meaning",
            "fluency",
            "coherence",
            "poeticness",
        ],
    },
    "yehia": {
        "dataset_id": "Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template",
        "split": "test",
        "poem_field": "generated_text",
        "description_field": "enhanced_description",
        "id_field_candidates": ["generation_id", "id", "input_row_id"],
        "judge_metrics": [
            "description_adherence",
            "meaning",
            "fluency",
            "coherence",
            "poeticness",
        ],
    },
    "fanar": {
        "dataset_id": "Shaer-AI/fanar-eval-native-prompt",
        "split": "test",
        "poem_field": "generated_text",
        "description_field": "enhanced_description",
        "id_field_candidates": ["generation_id", "id", "input_row_id"],
        "judge_metrics": [
            "meaning",
            "fluency",
            "coherence",
            "poeticness",
        ],
    },
}

DATASET_ORDER = ["shaer", "ashaar", "yehia", "fanar"]
