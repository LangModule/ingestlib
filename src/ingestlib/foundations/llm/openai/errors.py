"""Error translation for the failures OpenAI users actually hit.

The SDK's typed exceptions make detection exact, but their messages stop
short of the fix — and the worst one misleads: an exhausted balance
arrives as RateLimitError (code insufficient_quota), which reads as rate
limiting when it is billing. Callers re-raise with the hint, keeping the
original exception chained.
"""
import openai

from ingestlib.config import get_openai_config


def openai_error_hint(exc: Exception) -> str | None:
    """One-sentence fix for the classic failures, or None."""
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return (
            "OpenAI rejected the API key — check OPENAI_API_KEY in .env; "
            "manage keys at platform.openai.com/api-keys"
        )
    if isinstance(exc, openai.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota" or "insufficient_quota" in str(exc):
            return (
                "OpenAI account is out of credit (insufficient_quota) — this is "
                "billing, not rate limiting; add credit at platform.openai.com"
            )
        return "OpenAI rate limit hit and retries exhausted — wait a moment and retry"
    if isinstance(exc, openai.NotFoundError):
        cfg = get_openai_config()
        return (
            f"OpenAI does not recognize the model — check llm_model_id / "
            f"embedding_model_id in config.yaml's openai section "
            f"(current: {cfg.llm_model_id!r}, {cfg.embedding_model_id!r})"
        )
    if isinstance(exc, openai.APIConnectionError):
        return "could not reach api.openai.com — check your network/proxy"
    return None
