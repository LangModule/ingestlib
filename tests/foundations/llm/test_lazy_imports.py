"""The AWS import tax — pure, always run.

A pipeline configured for openai or ollama must never load boto3 or
langchain_aws just to import the operations: the shared types live in
llm/types.py and the dispatch surface resolves its bedrock pieces lazily.
Subprocesses give an honest cold-import measurement (this process has
long since imported everything).
"""
import subprocess
import sys

_PROBE = """\
import sys
import ingestlib.operations
import ingestlib.foundations.llm as llm

assert "boto3" not in sys.modules, "importing operations must not load boto3"
assert "langchain_aws" not in sys.modules, "importing operations must not load langchain_aws"

# touching a lazy attribute pulls bedrock in — on demand, not before
_ = llm.reset_clients
assert "boto3" in sys.modules, "lazy attribute access must resolve the real backend"
print("ok")
"""


def test_operations_import_without_aws_sdks():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_unknown_lazy_attribute_raises():
    import ingestlib.foundations.llm as llm

    try:
        llm.definitely_not_a_thing
    except AttributeError as exc:
        assert "definitely_not_a_thing" in str(exc)
    else:
        raise AssertionError("unknown attribute must raise AttributeError")
