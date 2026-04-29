from __future__ import annotations

from pathlib import Path
from typing import TextIO

from batch_label.client import LlamaServerClient
from batch_label.config import load_image_caption
from batch_label.discovery import captions_output_dir, sidecar_path_for
from batch_label.image_prep import USER_IMAGE_INSTRUCTION, prepare_image
from batch_label.messages import append_prompt_text
from batch_label.types import AppConfig, BatchResult, ImageJob


def build_jobs(image_paths: list[Path], images_root: Path) -> list[ImageJob]:
    output_dir = captions_output_dir(images_root)
    return [
        ImageJob(image_path=path, sidecar_path=sidecar_path_for(path, output_dir))
        for path in image_paths
    ]


def run_batch(
    config: AppConfig,
    jobs: list[ImageJob],
    client: LlamaServerClient,
    model_id: str,
    system_prompt: str,
    *,
    stream: TextIO,
) -> BatchResult:
    processed = 0
    skipped = 0
    failed = 0

    for job in jobs:
        if job.sidecar_path.exists() and not config.force:
            skipped += 1
            if config.verbose:
                print(f"Skipping {job.image_path}: sidecar already exists.", file=stream)
            continue

        try:
            prepared_image = prepare_image(job.image_path)
            image_caption = load_image_caption(job.image_path)
            effective_system_prompt = system_prompt
            if image_caption:
                effective_system_prompt = append_prompt_text(system_prompt, image_caption)

            label_response = client.label_image(
                model_id,
                effective_system_prompt,
                USER_IMAGE_INSTRUCTION,
                prepared_image,
            )
            job.sidecar_path.write_text(f"{label_response.text}\n", encoding="utf-8")
            processed += 1
            if config.verbose:
                print(
                    f"Labeled {job.image_path} -> {job.sidecar_path} "
                    f"({prepared_image.width}x{prepared_image.height})",
                    file=stream,
                )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"Failed {job.image_path}: {exc}", file=stream)

    print(
        f"Processed: {processed}, skipped: {skipped}, failed: {failed}",
        file=stream,
    )
    return BatchResult(processed=processed, skipped=skipped, failed=failed)
