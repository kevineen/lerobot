#!/usr/bin/env python

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.configs import parser
from lerobot.experiments.saycan_clip.clip_scorer import ClipScorer
from lerobot.experiments.saycan_clip.configs import SayCanClipRunConfig
from lerobot.experiments.saycan_clip.image_source import FrontRealSenseSource, load_images_from_paths
from lerobot.experiments.saycan_clip.llm_client import ChatMessage, OpenAICompatibleClient
from lerobot.experiments.saycan_clip.skills import SkillCandidate


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")


def _maybe_init_rerun(cfg: SayCanClipRunConfig) -> None:
    if not cfg.rerun:
        return
    rr.init("saycan_clip")
    if cfg.rerun_ip and cfg.rerun_port:
        try:
            rr.connect_grpc(url=f"rerun+http://{cfg.rerun_ip}:{cfg.rerun_port}/proxy")
        except Exception as e:
            logging.warning("Failed to connect to rerun server (%s). Continuing without rerun.", e)
    else:
        # Avoid spawning viewer in headless contexts; user can enable ip/port explicitly.
        raise ValueError("rerun is enabled but rerun_ip/rerun_port are not set.")


def _log_rerun(image_rgb: np.ndarray, ranked: list[SkillCandidate]) -> None:
    try:
        rr.log("camera/front", rr.Image(image_rgb))
        if ranked:
            summary = "\n".join([f"{i+1}. {c.raw_text}  (score={c.score:.4f})" for i, c in enumerate(ranked)])
            rr.log("saycan_clip/topk", rr.TextDocument(summary))
    except Exception as e:
        logging.warning("Rerun logging failed (%s). Continuing.", e)


def _parse_candidates_from_llm(text: str) -> list[str]:
    """Parse candidates from the LLM output.

    The prompt requests a JSON array of strings, but we accept a few common wrappers.
    """
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract the first JSON array substring.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON array from LLM output: {text[:200]!r}")


def _get_candidates(cfg: SayCanClipRunConfig) -> list[str]:
    if not cfg.saycan.llm_generate_candidates:
        return list(cfg.saycan.candidates)

    client = OpenAICompatibleClient(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        timeout_s=cfg.llm.timeout_s,
    )
    prompt = cfg.saycan.prompt_template.format(instruction=cfg.saycan.instruction, n=cfg.saycan.llm_num_candidates)
    out = client.chat_completions(
        model=cfg.llm.model,
        messages=[ChatMessage(role="user", content=prompt)],
    )
    return _parse_candidates_from_llm(out)


@parser.wrap()
def run(cfg: SayCanClipRunConfig) -> None:
    _init_logging()
    logging.info("Say-Can + CLIP run config:\n%s", pformat(asdict(cfg)))
    _maybe_init_rerun(cfg)

    scorer = ClipScorer(
        pretrained_name=cfg.clip.pretrained_name,
        device=cfg.clip.device,
        batch_size=cfg.clip.batch_size,
        normalize_embeddings=cfg.clip.normalize_embeddings,
    )

    # Setup image source
    if cfg.image_source.mode == "front_cam":
        if not cfg.image_source.front_cam.serial_number_or_name:
            raise ValueError("image_source.front_cam.serial_number_or_name must be set for front_cam mode.")
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        cam_cfg = RealSenseCameraConfig(
            serial_number_or_name=cfg.image_source.front_cam.serial_number_or_name,
            width=cfg.image_source.front_cam.width,
            height=cfg.image_source.front_cam.height,
            fps=cfg.image_source.front_cam.fps,
        )
        source = FrontRealSenseSource(cam_cfg)
        source.connect()
        images = None
    elif cfg.image_source.mode == "path_glob":
        images = load_images_from_paths(cfg.image_source.paths)
        if not images:
            raise ValueError("No images matched image_source.paths.")
        source = None
    else:
        raise ValueError(f"Unknown image_source.mode={cfg.image_source.mode!r}")

    try:
        for step in range(cfg.num_steps):
            if source is not None:
                img_meta = source.read()
            else:
                img_meta = images[min(step, len(images) - 1)]

            candidates = _get_candidates(cfg)
            scored = scorer.score(img_meta.image, candidates)
            top = scored[: max(0, cfg.saycan.top_k)]
            ranked = [SkillCandidate(name=c.text, score=c.score, raw_text=c.text) for c in top]

            print(f"\n--- Step {step+1}/{cfg.num_steps} | image={img_meta.source} ---")
            for i, c in enumerate(ranked):
                print(f"{i+1:02d}. score={c.score:.4f} | {c.raw_text}")

            if cfg.rerun:
                _log_rerun(img_meta.image, ranked)
    finally:
        if source is not None:
            source.disconnect()


def main() -> None:
    run()


if __name__ == "__main__":
    main()

