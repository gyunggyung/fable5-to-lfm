from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptBuild:
    prompt: str
    status: str
    error: str | None = None
    stripped_thinking_blocks: int = 0


THINK_BLOCK_RE = re.compile(r"<think>\s*.*?</think>\s*", re.DOTALL)
THOUGHT_CHANNEL_RE = re.compile(r"<\|channel\>thought\n.*?<channel\|>\s*", re.DOTALL)


def sanitize_name(value: str) -> str:
    return value.rstrip("/").split("/")[-1].replace(" ", "-")


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
        if normalized:
            return normalized
    return [{"role": "user", "content": str(row.get("prompt", ""))}]


def strip_thinking_blocks(content: str) -> tuple[str, int]:
    content, think_count = THINK_BLOCK_RE.subn("", content)
    content, channel_count = THOUGHT_CHANNEL_RE.subn("", content)
    return content.lstrip(), think_count + channel_count


def sanitize_history(messages: list[dict[str, str]], strip_thinking_history: bool = False) -> tuple[list[dict[str, str]], int]:
    if not strip_thinking_history:
        return messages, 0
    stripped = 0
    normalized: list[dict[str, str]] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            content, count = strip_thinking_blocks(content)
            stripped += count
        normalized.append({"role": message["role"], "content": content})
    return normalized, stripped


def render_chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = "assistant" if message["role"] == "assistant" else message["role"]
        if role == "tool":
            role = "user"
        parts.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_gemma4_turn(
    messages: list[dict[str, str]],
    enable_thinking: bool = False,
    empty_thought_channel: bool = False,
) -> str:
    parts: list[str] = ["<bos>"]
    if enable_thinking:
        parts.append("<|turn>system\n<|think|>\n<turn|>\n")
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        if role == "tool":
            role = "user"
        parts.append(f"<|turn>{role}\n")
        parts.append(f"{message['content'].strip()}<turn|>\n")
    parts.append("<|turn>model\n")
    if not enable_thinking and empty_thought_channel:
        parts.append("<|channel>thought\n<channel|>")
    return "".join(parts)


def render_deepseek_v4_chat(messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    user_token = "<｜User｜>"
    assistant_token = "<｜Assistant｜>"
    parts: list[str] = [bos]
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            parts.append(content)
        elif role == "user" or role == "tool":
            parts.append(f"{user_token}{content}")
        elif role == "assistant":
            parts.append(f"{assistant_token}{content}{eos}")
    parts.append(assistant_token)
    if enable_thinking:
        parts.append("<think>")
    else:
        parts.append("</think>")
    return "".join(parts)


def infer_fallback_style(model_name: str, tokenizer: Any) -> str | None:
    value = f"{model_name} {getattr(tokenizer, 'name_or_path', '')}".lower()
    if "gemma-4" in value:
        return "gemma4_turn"
    if "deepseek-v4" in value or "deepseek_v4" in value:
        return "deepseek_v4"
    if any(marker in value for marker in ("qwen", "lfm", "nemotron-terminal", "ouro")):
        return "chatml"
    return None


def build_prompt(
    tokenizer: Any,
    row: dict[str, Any],
    model_name: str = "",
    chat_template_kwargs: dict[str, Any] | None = None,
    strip_thinking_history: bool = False,
    gemma4_empty_thought_channel: bool = False,
    use_gemma4_patched_template: bool = False,
) -> PromptBuild:
    messages, stripped = sanitize_history(
        row_messages(row),
        strip_thinking_history=strip_thinking_history,
    )
    chat_template_kwargs = chat_template_kwargs or {}
    if use_gemma4_patched_template:
        return PromptBuild(
            prompt=render_gemma4_turn(
                messages,
                enable_thinking=bool(chat_template_kwargs.get("enable_thinking", False)),
                empty_thought_channel=gemma4_empty_thought_channel,
            ),
            status="gemma4_patched",
            stripped_thinking_blocks=stripped,
        )
    try:
        return PromptBuild(
            prompt=tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            ),
            status="chat_template",
            stripped_thinking_blocks=stripped,
        )
    except Exception as exc:
        style = infer_fallback_style(model_name, tokenizer)
        if style == "chatml":
            return PromptBuild(
                prompt=render_chatml(messages),
                status="chatml_fallback",
                error=str(exc),
                stripped_thinking_blocks=stripped,
            )
        if style == "gemma4_turn":
            return PromptBuild(
                prompt=render_gemma4_turn(
                    messages,
                    enable_thinking=bool(chat_template_kwargs.get("enable_thinking", False)),
                    empty_thought_channel=gemma4_empty_thought_channel,
                ),
                status="gemma4_fallback",
                error=str(exc),
                stripped_thinking_blocks=stripped,
            )
        if style == "deepseek_v4":
            return PromptBuild(
                prompt=render_deepseek_v4_chat(
                    messages,
                    enable_thinking=bool(chat_template_kwargs.get("enable_thinking", False)),
                ),
                status="deepseek_v4_fallback",
                error=str(exc),
                stripped_thinking_blocks=stripped,
            )
        return PromptBuild(
            prompt=str(row.get("prompt", "")),
            status="raw_fallback",
            error=str(exc),
            stripped_thinking_blocks=stripped,
        )


def build_prompts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    model_name: str = "",
    chat_template_kwargs: dict[str, Any] | None = None,
    prompt_options: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    chat_template_kwargs = chat_template_kwargs or {}
    prompt_options = prompt_options or {}
    builds = [
        build_prompt(
            tokenizer,
            row,
            model_name=model_name,
            chat_template_kwargs=chat_template_kwargs,
            strip_thinking_history=bool(prompt_options.get("strip_thinking_history", False)),
            gemma4_empty_thought_channel=bool(prompt_options.get("gemma4_empty_thought_channel", False)),
            use_gemma4_patched_template=bool(prompt_options.get("use_gemma4_patched_template", False)),
        )
        for row in rows
    ]
    counts = Counter(build.status for build in builds)
    errors = sorted({build.error for build in builds if build.error})[:3]
    rank_eligible = not counts.get("raw_fallback")
    status = "chat_template" if counts == {"chat_template": len(builds)} else "model_specific_or_mixed"
    if counts.get("raw_fallback"):
        status = "mixed_or_raw"
    return [build.prompt for build in builds], {
        "template_status_counts": dict(counts),
        "template_status": status,
        "rank_eligible": rank_eligible,
        "raw_fallback_errors": errors,
        "chat_template_kwargs": chat_template_kwargs,
        "prompt_options": prompt_options,
        "stripped_thinking_blocks": sum(build.stripped_thinking_blocks for build in builds),
    }
