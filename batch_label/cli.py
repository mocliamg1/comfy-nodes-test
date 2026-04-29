from __future__ import annotations

import argparse
import sys
from typing import Sequence, TextIO

from batch_label.client import LlamaServerClient, LlamaServerError, resolve_model_id
from batch_label.config import (
    BatchLabelError,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_PROMPT_FORMAT,
    build_config,
    load_prompt,
)
from batch_label.discovery import discover_images
from batch_label.pipeline import build_jobs, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch label images via llama-server.")
    parser.add_argument("--images", required=True, help="Input image path or directory.")
    parser.add_argument("--prompt-file", required=True, help="Global system prompt file.")
    parser.add_argument(
        "--prompt-format",
        choices=("auto", "text", "json"),
        default=DEFAULT_PROMPT_FORMAT,
        help="Prompt file format. Default: %(default)s",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base llama-server API URL. Default: %(default)s",
    )
    parser.add_argument("--model", help="Explicit model ID to use.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt sidecars.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recursively scan input directories (default).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top level of the input directory.",
    )
    parser.add_argument(
        "--extensions",
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress and HTTP summaries.",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, stderr: TextIO | None = None) -> int:
    parser = build_parser()
    output_stream = stderr or sys.stderr

    try:
        config = build_config(parser.parse_args(argv))
        prompt = load_prompt(config.prompt_file, config.prompt_format)
        image_paths = discover_images(config.images_path, config.extensions, config.recursive)
        if not image_paths:
            raise BatchLabelError(
                "No images found for the provided path and extensions: "
                + ", ".join(config.extensions)
            )

        jobs = build_jobs(image_paths, config.images_path)
        with LlamaServerClient(config.base_url, config.timeout) as client:
            model_id = resolve_model_id(client, config.model)
            result = run_batch(
                config=config,
                jobs=jobs,
                client=client,
                model_id=model_id,
                system_prompt=prompt,
                stream=output_stream,
            )
        return 0 if result.failed == 0 else 1
    except (BatchLabelError, LlamaServerError) as exc:
        print(f"Error: {exc}", file=output_stream)
        return 1


def run() -> int:
    return main()
