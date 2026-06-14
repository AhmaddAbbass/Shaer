#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

from common import load_env
from model_registry import ModelSpec


def choose_torch_dtype() -> Any:
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class TransformersGenerator:
    def __init__(self, spec: ModelSpec, args: Any, logger: Any) -> None:
        load_env()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.args = args
        self.logger = logger
        self.torch = torch
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
            "low_cpu_mem_usage": True,
            "token": os.getenv("HF_TOKEN") or None,
        }
        model_kwargs["torch_dtype"] = choose_torch_dtype()
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
            token=os.getenv("HF_TOKEN") or None,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = getattr(self.model, "device", None)
        self.logger.info("loaded transformers model=%s model_id=%s", spec.name, spec.model_id)

    def _render_prompt(self, prompt: str) -> str:
        if not getattr(self.args, "use_chat_template", False):
            return prompt
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        messages = [
            {"role": "system", "content": "أخرج القصيدة العربية فقط دون شرح أو تعليق."},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.spec.name == "yehia_base":
            kwargs["add_generation_prompt"] = True
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        rendered_prompts = [self._render_prompt(prompt) for prompt in prompts]
        inputs = self.tokenizer(rendered_prompts, return_tensors="pt", padding=True)
        if self.device is not None:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_width = int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=bool(float(self.args.temperature) > 0),
                temperature=float(self.args.temperature),
                top_p=float(self.args.top_p),
                repetition_penalty=float(self.args.repetition_penalty),
                max_new_tokens=int(self.args.max_new_tokens),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
            )
        texts = []
        for sequence in output:
            new_tokens = sequence[prompt_width:]
            if self.spec.name == "ashaar_model":
                texts.append(self._decode_ashaar_tokens(new_tokens).strip())
            else:
                texts.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return texts

    def _decode_ashaar_tokens(self, token_ids: Any) -> str:
        text = ""
        for token_id in token_ids:
            decoded = self.tokenizer.decode(token_id, skip_special_tokens=False)
            if "meter" in decoded or "theme" in decoded:
                break
            if decoded == "<|vsep|>":
                text += "\n\n"
            elif decoded in {"<|bsep|>", "</|bsep|>"}:
                text += "\n"
            elif decoded in {"<|psep|>", "</|psep|>"}:
                continue
            else:
                text += decoded
        return text


class LlamaCppGenerator:
    def __init__(self, spec: ModelSpec, args: Any, logger: Any) -> None:
        load_env()
        from huggingface_hub import hf_hub_download, list_repo_files
        from llama_cpp import Llama

        self.spec = spec
        self.args = args
        self.logger = logger
        hf_token = os.getenv("HF_TOKEN") or None
        gguf_filename = str(getattr(args, "gguf_filename", "") or "").strip()
        if not gguf_filename:
            repo_files = list_repo_files(spec.model_id, token=hf_token)
            candidates = [name for name in repo_files if name.lower().endswith(".gguf")]
            preferred_suffixes = ("Q4_K_M.gguf", "Q5_K_M.gguf", "Q8_0.gguf", ".gguf")
            for suffix in preferred_suffixes:
                match = next((name for name in candidates if name.endswith(suffix)), "")
                if match:
                    gguf_filename = match
                    break
        if not gguf_filename:
            raise RuntimeError(
                f"Could not infer a GGUF filename for {spec.model_id}. Pass --gguf-filename explicitly."
            )
        gguf_path = hf_hub_download(repo_id=spec.model_id, filename=gguf_filename, token=hf_token)
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=int(args.ctx_size),
            n_gpu_layers=int(args.gpu_layers),
            verbose=False,
        )
        self.logger.info("loaded llama_cpp model=%s model_id=%s gguf=%s", spec.name, spec.model_id, gguf_filename)

    def generate(self, prompt: str) -> str:
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=int(self.args.max_new_tokens),
            temperature=float(self.args.temperature),
            top_p=float(self.args.top_p),
            repeat_penalty=float(self.args.repetition_penalty),
        )
        return str(output["choices"][0]["text"]).strip()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]


def build_generator(spec: ModelSpec, args: Any, logger: Any) -> Any:
    if spec.backend == "transformers":
        return TransformersGenerator(spec, args, logger)
    if spec.backend == "llama_cpp":
        return LlamaCppGenerator(spec, args, logger)
    raise RuntimeError(f"Unsupported backend '{spec.backend}' for model '{spec.name}'.")
