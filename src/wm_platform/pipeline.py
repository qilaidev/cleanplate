from __future__ import annotations

from wm_platform.config import Settings

DEFAULT_WATERMARK_PROMPT = (
    "watermark, logo, on-screen text overlay, corner badge, platform stamp"
)

AI_INPAINT_PROVIDERS = ("minimax_remover", "comfy_diffueraser")
PROBE_ORDER = ("minimax_remover", "comfy_diffueraser", "local_fallback")


def auto_inpaint_chain(settings: Settings) -> list[str]:
    chain = ["minimax_remover", "comfy_diffueraser"]
    if settings.auto_allow_copy_fallback:
        chain.append("local_fallback")
    return chain
