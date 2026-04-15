from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMBackendConfig:
    """OpenAI-compatible backend config (vLLM / LM Studio)."""

    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str | None = None
    model: str = "local-model"
    timeout_s: float = 30.0


@dataclass
class CLIPConfig:
    """CLIP scoring configuration."""

    pretrained_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    batch_size: int = 16
    normalize_embeddings: bool = True


@dataclass
class FrontCamConfig:
    """Front camera acquisition config (RealSense by default)."""

    # For RealSense in LeRobot, the camera type choice name is "intelrealsense".
    serial_number_or_name: str = ""
    width: int | None = None
    height: int | None = None
    fps: int | None = None


@dataclass
class ImageSourceConfig:
    """Image source for CLIP scoring."""

    # front_cam: reads a single RGB frame from a connected robot camera
    # path_glob: reads images from filesystem
    mode: str = "front_cam"  # "front_cam" | "path_glob"

    # used when mode == "front_cam"
    front_cam_name: str = "front"
    front_cam: FrontCamConfig = field(default_factory=FrontCamConfig)

    # used when mode == "path_glob"
    paths: list[str] = field(default_factory=list)


@dataclass
class SayCanConfig:
    """Say-Can-style candidate generation config.

    Phase A uses either a fixed candidate list or an LLM-generated list.
    """

    instruction: str = "pick up the object"
    candidates: list[str] = field(
        default_factory=lambda: [
            "pick up the object",
            "move closer to the object",
            "open the gripper",
            "close the gripper",
            "place the object on the table",
        ]
    )
    top_k: int = 5

    llm_generate_candidates: bool = False
    llm_num_candidates: int = 10
    prompt_template: str = (
        "You are helping a robot. Given an instruction, propose a list of short candidate actions.\n"
        "Instruction: {instruction}\n"
        "Return {n} actions as a JSON array of strings."
    )


@dataclass
class SayCanClipRunConfig:
    image_source: ImageSourceConfig = field(default_factory=ImageSourceConfig)
    saycan: SayCanConfig = field(default_factory=SayCanConfig)
    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)

    # Optional rerun logging (off by default)
    rerun: bool = False
    rerun_ip: str | None = None
    rerun_port: int | None = None

    # Looping behavior
    num_steps: int = 1

