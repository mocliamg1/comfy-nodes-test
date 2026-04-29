from __future__ import annotations


def append_prompt_text(base_text: str, extra_text: str) -> str:
    extra_text = extra_text.strip()
    if not extra_text:
        return base_text
    return f"{base_text.rstrip()}\n\n{extra_text}"


def format_batch_text_output(texts: list[str]) -> str:
    if len(texts) == 1:
        return texts[0]

    return "\n\n".join(
        f"Image {index}:\n{text}"
        for index, text in enumerate(texts, start=1)
    )
