"""remove() — one document erased from both stores. Always run, no gates."""
import pytest

from ingestlib.services import remove
from ingestlib.storage import artifacts

from tests.services.conftest import vec

_DOC_A = "remove-a-" + "0" * 55
_DOC_B = "remove-b-" + "1" * 55


def test_remove_by_path_erases_both_stores(stack, make_document):
    path = stack.corpus / "report.pdf"
    make_document(_DOC_A, path)
    assert stack.store.query(vec(1.0), top_k=5) != []

    result = remove(path, store=stack.store)

    assert result.doc_id == _DOC_A
    assert result.vectors_deleted == 1
    assert result.artifacts_deleted >= 3  # parse json + md + page + meta + split
    assert artifacts.document_exists(_DOC_A) is False
    assert stack.store.query(vec(1.0), top_k=5) == []


def test_remove_by_full_doc_id(stack, make_document):
    make_document(_DOC_A, stack.corpus / "report.pdf")
    result = remove(_DOC_A, store=stack.store)
    assert result.doc_id == _DOC_A
    assert artifacts.document_exists(_DOC_A) is False


def test_remove_by_unique_prefix(stack, make_document):
    """`ingestlib list` prints short ids — a unique prefix must resolve."""
    make_document(_DOC_A, stack.corpus / "a.pdf")
    make_document(_DOC_B, stack.corpus / "b.pdf")
    result = remove("remove-a", store=stack.store)
    assert result.doc_id == _DOC_A
    assert artifacts.document_exists(_DOC_B) is True, "only the match goes"


def test_ambiguous_prefix_refuses(stack, make_document):
    make_document(_DOC_A, stack.corpus / "a.pdf")
    make_document(_DOC_B, stack.corpus / "b.pdf")
    with pytest.raises(ValueError, match="ambiguous.*2"):
        remove("remove-", store=stack.store)
    assert artifacts.document_exists(_DOC_A) is True, "nothing may be deleted"


def test_unknown_target_names_the_fix(stack):
    with pytest.raises(ValueError, match="no stored document.*list"):
        remove("never-stored.pdf", store=stack.store)


def test_parsed_but_never_ingested_deletes_artifacts_only(stack, make_document):
    """No manifest → no vector deletion attempted, artifacts still erased."""
    make_document(_DOC_A, stack.corpus / "draft.pdf", with_vectors=False)
    result = remove(_DOC_A, store=stack.store)
    assert result.vectors_deleted == 0
    assert result.artifacts_deleted > 0
    assert artifacts.document_exists(_DOC_A) is False


def test_vector_deletion_uses_the_manifest_namespace(stack, make_document):
    """The manifest records where vectors actually went — deletion follows it,
    and path resolution is namespace-scoped."""
    path = stack.corpus / "tenant-doc.pdf"
    make_document(_DOC_A, path, namespace="tenant-a")

    with pytest.raises(ValueError, match="no stored document"):
        remove(path, store=stack.store)  # default namespace — not this doc

    result = remove(path, namespace="tenant-a", store=stack.store)
    assert result.vectors_deleted == 1
    assert stack.store.query(vec(1.0), top_k=5, namespace="tenant-a") == []
