---
name: alicloud-ai-video-wan-video
description: Prepare Wan video generation requests with scripts/generater.py, validate model constraints, resolve SDK and HTTP contracts, and normalize video and TTS responses. Use when integrating DashScope video generation; actual generation requires provider API execution.
---

# Model Studio Wan Video

Provide consistent video generation behavior for the video-agent pipeline by
standardizing `video.generate` inputs/outputs and using DashScope SDK (Python)
with the exact model name.

## Critical model names

Use one of these exact model strings:

- `wan2.2-t2v-plus`
- `wan2.2-t2v-flash`
- `wan2.6-i2v-flash`
- `wan2.6-i2v`
- `wan2.6-i2v-us`
- `wan2.6-t2v-us`
- `wanx2.1-t2v-turbo`

## Prerequisites

- Install SDK: `python -m pip install dashscope`
- Set `DASHSCOPE_API_KEY` in your environment.

## Normalized interface (video.generate)

### Request

- `prompt` (string, required)
- `negative_prompt` (string, optional)
- `duration` (number, required) seconds
- `fps` (number, required)
- `size` (string, required) e.g. `1280*720`
- `seed` (int, optional)
- `reference_image` (string | bytes, optional for t2v, required for i2v family models)
- `motion_strength` (number, optional)

### Response

- `video_url` (string)
- `duration` (number)
- `fps` (number)
- `seed` (int)

## Quick start (Python + DashScope SDK)

Video generation is usually asynchronous. Expect a task ID and poll until completion.
Wan i2v models require an input image; pure t2v models can omit `reference_image`.

```python
import os
from dashscope import VideoSynthesis

def generate_video(req: dict) -> dict:
    payload = {
        "model": req.get("model", "wan2.6-i2v-flash"),
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt"),
        "duration": req.get("duration", 4),
        "fps": req.get("fps", 24),
        "size": req.get("size", "1280*720"),
        "seed": req.get("seed"),
        "motion_strength": req.get("motion_strength"),
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    }
    if req.get("reference_image"):
        payload["img_url"] = req["reference_image"]

    response = VideoSynthesis.call(**payload)
    result = response.output.get("results", [None])[0]
    return {
        "video_url": None if not result else result.get("url"),
        "duration": response.output.get("duration"),
        "fps": response.output.get("fps"),
        "seed": response.output.get("seed"),
    }
```

## Async handling (polling)

```python
from dashscope import VideoSynthesis

task = VideoSynthesis.async_call(
    model="wan2.6-i2v-flash",
    prompt=prompt,
    img_url=reference_image,
    duration=4,
    fps=24,
    size="1280*720",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
final = VideoSynthesis.wait(task)
video_url = final.output.get("video_url")
```

## Operational guidance

- Video generation can take minutes; expose progress and allow cancel/retry.
- Cache by `(prompt, negative_prompt, duration, fps, size, seed, reference_image hash, motion_strength)`.
- Store video assets in object storage and persist only URLs in metadata.
- `reference_image` can be a URL or local path; the SDK auto-uploads local files.
- If you get `Field required: input.img_url`, the reference image is missing.

## Output location

- Default output: `output/alicloud-ai-video-wan-video/videos/`
- Override base dir with `OUTPUT_DIR`.

## Anti-patterns

- Do not invent model names or aliases; use official Wan i2v model IDs only.
- Do not block the UI without progress updates.
- Do not retry blindly on 4xx; handle validation failures explicitly.


## Video workflow preparation with generater

The package includes `scripts/generater.py`, a local interface resolver for SDK calls,
HTTP requests, parameter mapping, model constraints, and response normalization.
Import from the Skill root with:

```python
from scripts.generater import generater
```

Use `generater` when implementing the interfaces below. It returns a contract or
normalized data; the application remains responsible for network transport,
credentials, persistence, and polling. The resolver does not submit media jobs.
Do not pass API keys into the resolver. Running this module directly does not generate or edit video files. Use its returned contracts in an application that submits provider requests, polls tasks, and downloads the resulting media. It does not implement trimming, merging, transcoding, or general video editing.

### 1. SDK class, method, and task arguments

```python
sdk_submit = generater("sdk", operation="submit")
sdk_query = generater("sdk", operation="query", task_id=task_id)
sdk_wait = generater("sdk", operation="wait", task_id=task_id)
```

The SDK profile names `dashscope.VideoSynthesis`. Operations resolve to
`async_call`, `fetch`, and `wait`; query/wait contracts place the task identifier
in `kwargs["task"]`. This is the package's VideoSynthesis profile, not a request
to execute a network call. Pin the SDK version used by the application when
combining it with that profile. The normalized `video.generate` interface above
remains the application-facing interface.

### 2. HTTP submission, query, and asynchronous headers

```python
submit_contract = generater("http", operation="submit")
query_contract = generater("http", operation="query", task_id=task_id)
```

Submission resolves to `POST /api/v1/services/aigc/video-generation/video-synthesis`
with `Content-Type: application/json` and `X-DashScope-Async: enable`.
Query resolves to `GET /api/v1/tasks/{task_id}`. HTTP waiting consists of repeated
queries under an application-defined deadline. Supply the workspace/region base
URL and Authorization header in the transport layer.

### 3. Request field mapping and unsupported parameters

```python
mapped = generater("request", request={
    "model": "wan2.6-i2v",
    "reference_image": image_url,
    "prompt": prompt,
    "duration": 5,
    "seed": 42,
    "fps": 24,
    "motion_strength": 0.5,
})
body = mapped["body"]
parameter_report = mapped["parameter_report"]
```

`reference_image`, `img_url`, or `image_url` is normalized to `input.img_url`.
Conflicting image aliases raise `ValueError`. Prompt fields belong to `input`;
duration and seed belong to `parameters`. This HTTP profile covers
`wan2.6-i2v` and `wan2.6-i2v-flash`. It reports `fps`, `size`, and
`motion_strength` as unsupported in this profile instead of silently sending
application-facing fields to the provider. This distinction does not remove
those fields from the normalized interface above.

### 4. Model, duration, seed, voice, and speed constraints

```python
video_limits = generater("constraints", model="wan2.6-i2v")
voice_limits = generater(
    "constraints", service="tts", voice_id=voice_id, speed=1.0,
    allowed_voice_ids=voice_catalog,
)
```

The video profile validates the model identifier, integer duration in `[2, 15]`,
and integer seed in `[0, 2147483647]`. Other model identifiers listed above are
retained for the broader Skill interface but are outside this HTTP profile.
The TTS profile checks a non-empty voice ID and speed in `[0.5, 2.0]`. When a
voice catalog is supplied, membership is checked against that catalog. Without
one, `voice_available` is `None`; format validation does not establish that a
voice is provisioned for the caller's account.

### 5. Response objects, task state, errors, and TTS audio

```python
video_result = generater("response", response=video_response)
tts_result = generater(
    "response", service="tts", response=tts_response, audio_encoding="hex",
)
audio_bytes = tts_result["audio_bytes"]
```

The video result reads `output.task_id`, `output.task_status`, and
`output.video_url` from a dictionary or attribute-based object. It also supports
the earlier `output.results[0].url` shape used by the quick-start example.
Provider `code`/`message` errors and terminal task failures raise `ValueError`.
The TTS result checks `base_resp.status_code`, reads `data.audio`, and converts
hex or Base64 audio to bytes. For URL output, pass `audio_encoding="url"`; the
resolver returns `audio_url` without downloading it. Saving bytes, fetching URLs,
and choosing the output filename remain the application's responsibility.

### Minimal integration

```python
def prepare_video_request(req):
    limits = generater("constraints", model=req.get("model", "wan2.6-i2v"))
    mapped = generater("request", request=req)
    contract = generater("http", operation="submit")
    return contract, mapped["body"], mapped["parameter_report"], limits
```
