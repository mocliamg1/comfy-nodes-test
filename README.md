# Batch Label

Batch label images by sending one fresh multimodal chat request per image to an already-running
`llama-server`.

## Features

- Resizes images down to at most 1 megapixel before sending them to the API
- Sends the global prompt file as the `system` message on every request
- Sends each image in a brand-new chat with no shared history
- Writes one `.txt` sidecar next to each image
- Skips existing sidecars by default for resumable runs
- Accepts plain-text prompt files or JSON prompt files with a top-level `caption` key
- Automatically appends per-image instruction files like `photo.png.json` when present

## Requirements

- Python 3.9+
- A running `llama-server` instance with a vision-capable model

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

Start `llama-server` separately, for example:

```bash
llama-server -m /path/to/model.gguf --port 8080
```

Then run the labeler:

```bash
python3 -m batch_label \
  --images /path/to/images \
  --prompt-file /path/to/prompt.txt \
  --base-url http://127.0.0.1:8080/v1
```

To use a JSON instruction file:

```bash
python3 -m batch_label \
  --images /path/to/images \
  --prompt-file /path/to/prompt.json \
  --prompt-format json \
  --base-url http://127.0.0.1:8080/v1
```

If the server exposes multiple models, pass one explicitly:

```bash
python3 -m batch_label \
  --images /path/to/images \
  --prompt-file /path/to/prompt.txt \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen-3.5-vl
```

## ComfyUI

Install this repo into `ComfyUI/custom_nodes/BatchLabel`, then install its runtime dependencies:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone <this-repo-url> BatchLabel
cd BatchLabel
pip install -r requirements.txt
```

The custom node appears as `BatchLabel Llama.cpp Vision Text` under `BatchLabel/Llama.cpp`.

### Node Inputs

- `image`: ComfyUI `IMAGE` input, including batched images
- `system_prompt_preset`: one of:
  - `short_caption`
  - `detailed_description`
  - `ocr_text_and_scene`
  - `diffusion_json_blueprint`
  - `nano_banana_json_blueprint`
- `system_prompt_custom`: optional extra system prompt text appended to the preset
- `instructions`: optional extra user instructions appended to the image request
- `base_url`: `llama-server` OpenAI-compatible base URL, default `http://127.0.0.1:8080/v1`
- `model`: optional explicit model ID
- `timeout`: request timeout in seconds

### Node Behavior

- Sends one fresh `llama-server` chat request per input image
- Reuses the same image resizing logic as the CLI, capped at 1 megapixel
- Returns plain text only
- Uses the selected preset as the base system prompt and appends `system_prompt_custom` when present
- Uses the built-in image instruction as the base user text and appends `instructions` when present
- When the `IMAGE` input contains multiple images, the node returns one `STRING` formatted like:

```text
Image 1:
...

Image 2:
...
```

After the node finishes, it best-effort requests `llama.cpp` model unload and calls ComfyUI
`POST /free` with `{"unload_models": true, "free_memory": false}` to unload ComfyUI models from
VRAM.

## Prompt File

Plain-text prompt files are sent verbatim as the `system` prompt. JSON prompt files must contain a
top-level `caption` string, which is used as the `system` prompt.

If an image has a sibling file named like `photo.png.json`, its `caption` string is appended to the
system prompt for that image only.

## Output

For an input folder like `/data/images`, the tool writes all caption files into that folder. Top-level
images use `<stem>.txt`, while nested images are flattened into unique names like
`nested__photo.txt`. When the input is a single image path, the caption is written into that image's
parent folder. Each file contains only the final label text plus a trailing newline.
