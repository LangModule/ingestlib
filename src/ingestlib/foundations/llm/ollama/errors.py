"""Error translation for the failures every Ollama user hits.

A local server has no auth, quota, or rate limits — its failure modes are
"the server is not running" and "the model is not pulled", and both have
one-command fixes the error should hand over. The caller passes the model
it was using, so a user who pulled the chat model but not the embedder is
told exactly which one is missing. Callers re-raise with the hint, keeping
the original exception chained.
"""
import openai

from ingestlib.config import get_ollama_config


def ollama_error_hint(exc: Exception, model: str) -> str | None:
    """One-sentence fix for the classic failures, or None."""
    if isinstance(exc, openai.APIConnectionError):
        base_url = get_ollama_config().base_url
        return (
            f"cannot reach the Ollama server at {base_url} — start Ollama "
            f"(`ollama serve`, or open the app); using vLLM/LM Studio instead, "
            f"check ollama.base_url in config.yaml"
        )
    # Ollama reports a missing model as 404; some versions say 400 with
    # "not found" in the body.
    if isinstance(exc, openai.NotFoundError) or (
        isinstance(exc, openai.BadRequestError) and "not found" in str(exc).lower()
    ):
        return f"model {model!r} is not on the server — run `ollama pull {model}`"
    return None
