#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    model_id: str
    group: str
    backend: str
    role: str
    paper_source: str
    supports_generation: bool = True
    default_prompt_mode: str = "plain"
    notes: str = ""
    generation_defaults: dict[str, object] = field(default_factory=dict)


MODEL_SPECS = {
    "yehia_base": ModelSpec(
        name="yehia_base",
        display_name="Original Yehia",
        model_id="Navid-AI/Yehia-7B-preview",
        group="instruction",
        backend="transformers",
        role="Your baseline/original model",
        paper_source="Model card for Navid-AI/Yehia-7B-preview",
        default_prompt_mode="plain_instruction",
        notes="Gated on Hugging Face; access must be accepted before download.",
    ),
    "ashaar_model": ModelSpec(
        name="ashaar_model",
        display_name="Ashaar",
        model_id="arbml/Ashaar_model",
        group="continuation",
        backend="transformers",
        role="Poetry generation / continuation",
        paper_source="Ashaar: Automatic Analysis and Generation of Arabic Poetry Using Deep Learning Approaches",
        default_prompt_mode="continuation_prefix",
    ),
    "fanar_2_diwan": ModelSpec(
        name="fanar_2_diwan",
        display_name="Fanar 2 Diwan",
        model_id="QCRI/Fanar-2-Diwan",
        group="instruction",
        backend="transformers",
        role="Optional strong Arabic LLM",
        paper_source="Fanar: An Arabic-Centric Multimodal Generative AI Platform",
        default_prompt_mode="fanar_metadata",
        notes="Native prompting expects structured metadata fields such as topic, era, poet id, and rhyme letter.",
    ),
    "fanar_2_diwan_prefix": ModelSpec(
        name="fanar_2_diwan_prefix",
        display_name="Fanar 2 Diwan (Prefix Continuation)",
        model_id="QCRI/Fanar-2-Diwan",
        group="continuation",
        backend="transformers",
        role="Experimental continuation baseline",
        paper_source="Fanar 2.0: Arabic Generative AI Stack",
        default_prompt_mode="continuation_prefix",
        notes="Experimental use with Shaer first-bayt prefixes. Native Fanar prompting expects structured metadata such as topic, era, poet id, and rhyme letter.",
    ),
    "gpt2_small_arabic_poetry": ModelSpec(
        name="gpt2_small_arabic_poetry",
        display_name="GPT-2 Small Arabic Poetry",
        model_id="akhooli/gpt2-small-arabic-poetry",
        group="continuation",
        backend="transformers",
        role="GPT-2 Arabic poetry generator",
        paper_source="Related base work: AraGPT2",
        default_prompt_mode="continuation_prefix",
    ),
    "gpt2_medium_arabic_poetry": ModelSpec(
        name="gpt2_medium_arabic_poetry",
        display_name="GPT-2 Medium Arabic Poetry",
        model_id="elgeish/gpt2-medium-arabic-poetry",
        group="continuation",
        backend="transformers",
        role="GPT-2 Arabic poetry generator",
        paper_source="Related base work: AraGPT2 / GPT-2 Arabic generation literature",
        default_prompt_mode="continuation_prefix",
    ),
    "qwen3_poetry_gguf": ModelSpec(
        name="qwen3_poetry_gguf",
        display_name="Qwen3 Poetry GGUF",
        model_id="Cyb3RQ/arabic-poetry-qwen3-8b-GGUF",
        group="instruction",
        backend="llama_cpp",
        role="Qwen3-8B poetry fine-tune",
        paper_source="Base model paper: Qwen3 Technical Report",
        default_prompt_mode="plain_instruction",
        notes="Requires llama-cpp-python and a GGUF filename.",
    ),
    "arapoembert": ModelSpec(
        name="arapoembert",
        display_name="AraPoemBERT",
        model_id="faisalq/bert-base-arapoembert",
        group="analysis",
        backend="transformers",
        role="Optional evaluator / analysis model, not continuation",
        paper_source="AraPoemBERT: A Pretrained Language Model for Arabic Poetry Analysis",
        supports_generation=False,
        notes="Not a generation model; exclude from generation suites.",
    ),
}


ALIASES = {
    "yehia": "yehia_base",
    "original_yehia": "yehia_base",
    "ashaar": "ashaar_model",
    "fanar": "fanar_2_diwan",
    "fanar2": "fanar_2_diwan",
    "fanar_prefix": "fanar_2_diwan_prefix",
    "fanar_diwan_prefix": "fanar_2_diwan_prefix",
    "gpt2_small": "gpt2_small_arabic_poetry",
    "gpt2_medium": "gpt2_medium_arabic_poetry",
    "qwen3_gguf": "qwen3_poetry_gguf",
    "qwen3_poetry": "qwen3_poetry_gguf",
    "bert_arapoembert": "arapoembert",
}


def get_model_spec(name: str) -> ModelSpec:
    key = str(name or "").strip()
    key = ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise KeyError(f"Unknown model '{name}'. Supported models: {supported}")
    return MODEL_SPECS[key]


def parse_model_list(csv_value: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for raw_name in str(csv_value or "").split(","):
        name = raw_name.strip()
        if not name:
            continue
        spec = get_model_spec(name)
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)
    return specs
