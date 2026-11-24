"""Local interface contracts and response normalization for the video adapter."""
from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

_CALL_EVENTS: list[str] = []
_I2V_MODELS = ("wan2.6-i2v", "wan2.6-i2v-flash")


def _field(value: Any, key: str, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _model(model):
    if model not in _I2V_MODELS:
        raise ValueError("model is outside the Wan 2.6 i2v HTTP profile")
    return model


def _video_limits(model):
    return {"model": _model(model), "duration_min": 2, "duration_max": 15,
            "seed_min": 0, "seed_max": 2147483647,
            "unsupported_fields": ["fps", "size", "motion_strength"]}


def generater(area: str, **context) -> dict:
    """Resolve sdk/http/request/constraints/response without sending requests."""
    _CALL_EVENTS.append(area if isinstance(area, str) else "invalid")
    if area == "sdk":
        operation = context.get("operation", "submit")
        methods = {"submit": "async_call", "query": "fetch", "wait": "wait"}
        if operation not in methods:
            raise ValueError("SDK operation must be submit, query, or wait")
        kwargs = {}
        if operation != "submit":
            task_id = context.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("task_id is required")
            kwargs["task"] = task_id
        return {"class_name": "dashscope.VideoSynthesis", "method": methods[operation], "kwargs": kwargs}
    if area == "http":
        operation = context.get("operation", "submit")
        if operation == "submit":
            return {"method": "POST", "path": "/api/v1/services/aigc/video-generation/video-synthesis",
                    "headers": {"Content-Type": "application/json", "X-DashScope-Async": "enable"}}
        if operation == "query":
            task_id = context.get("task_id")
            if not isinstance(task_id, str) or not task_id or any(c in task_id for c in "/?#"):
                raise ValueError("task_id must be a single path segment")
            return {"method": "GET", "path": "/api/v1/tasks/" + task_id, "headers": {}}
        raise ValueError("HTTP operation must be submit or query")
    if area == "request":
        req = context.get("request")
        if not isinstance(req, Mapping):
            raise ValueError("request must be a mapping")
        model = _model(req.get("model", "wan2.6-i2v"))
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        images = [req[k] for k in ("reference_image", "img_url", "image_url") if req.get(k) is not None]
        if not images or any(image != images[0] for image in images):
            raise ValueError("one consistent reference image is required")
        image = images[0]
        if not isinstance(image, str) or not image.startswith(("http://", "https://")):
            raise ValueError("this HTTP profile requires a public image URL")
        duration = req.get("duration", 5)
        if type(duration) is not int or not 2 <= duration <= 15:
            raise ValueError("duration must be an integer in [2,15]")
        parameters = {"duration": duration}
        if req.get("seed") is not None:
            seed = req["seed"]
            if type(seed) is not int or not 0 <= seed <= 2147483647:
                raise ValueError("seed must be an integer in [0,2147483647]")
            parameters["seed"] = seed
        input_data = {"prompt": prompt, "img_url": image}
        if req.get("negative_prompt") is not None:
            input_data["negative_prompt"] = req["negative_prompt"]
        report = {key: "unsupported" for key in ("fps", "size", "motion_strength") if key in req}
        report.update({key: "applied" for key in ("prompt", "duration", "seed") if key in req})
        report["reference_image"] = "mapped_to_input.img_url"
        return {"body": {"model": model, "input": input_data, "parameters": parameters}, "parameter_report": report}
    if area == "constraints":
        if context.get("service", "video") == "video":
            return _video_limits(context.get("model", "wan2.6-i2v"))
        if context.get("service") != "tts":
            raise ValueError("service must be video or tts")
        voice = context.get("voice_id")
        speed = context.get("speed", 1.0)
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("voice_id is required")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.5 <= speed <= 2:
            raise ValueError("speed must be in [0.5,2.0]")
        catalog = context.get("allowed_voice_ids")
        available = None if catalog is None else voice in catalog
        if available is False:
            raise ValueError("voice_id is not in the supplied catalog")
        return {"voice_id": voice, "speed": speed, "voice_available": available}
    if area == "response":
        response = context.get("response")
        if response is None:
            raise ValueError("response is required")
        if context.get("service", "video") == "tts":
            base = _field(response, "base_resp", {})
            if _field(base, "status_code", 0) not in (0, None):
                raise ValueError(str(_field(base, "status_msg", "TTS request failed")))
            audio = _field(_field(response, "data", {}), "audio")
            if not isinstance(audio, str):
                raise ValueError("data.audio is missing")
            encoding = context.get("audio_encoding", "hex")
            if encoding == "url":
                return {"audio_url": audio, "audio_bytes": None}
            if encoding == "hex":
                data = bytes.fromhex(audio)
            elif encoding == "base64":
                data = base64.b64decode(audio, validate=True)
            else:
                raise ValueError("audio_encoding must be hex, base64, or url")
            return {"audio_bytes": data, "audio_url": None}
        if context.get("service", "video") != "video":
            raise ValueError("service must be video or tts")
        code = _field(response, "code")
        if code:
            raise ValueError(str(code) + ": " + str(_field(response, "message", "")))
        status_code = _field(response, "status_code")
        if isinstance(status_code, int) and not 200 <= status_code < 300:
            raise ValueError("HTTP status " + str(status_code))
        output = _field(response, "output", {})
        status = _field(output, "task_status")
        if status in ("FAILED", "CANCELED"):
            raise ValueError(str(_field(output, "code", status)) + ": " + str(_field(output, "message", "")))
        url = _field(output, "video_url")
        results = _field(output, "results", [])
        if not url and results:
            url = _field(results[0], "url")
        return {"task_id": _field(output, "task_id"), "task_status": status, "video_url": url,
                "duration": _field(output, "duration"), "fps": _field(output, "fps"), "seed": _field(output, "seed")}
    raise ValueError("area must be sdk, http, request, constraints, or response")
