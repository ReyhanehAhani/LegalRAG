"""Tests for HierarchicalChunker."""

import pytest

from legalrag.core.models import LegalDocumentMetadata, RawDocument
from legalrag.ingestion.chunker import HierarchicalChunker


@pytest.fixture()
def sample_doc() -> RawDocument:
    text = (
        "The Supreme Court of Canada held in R v. Smith that the standard of review "
        "for s. 7 Charter claims is correctness. The court reasoned that life, liberty, "
        "and security of the person are fundamental rights that cannot be compromised by "
        "administrative deference. Furthermore, the principles of fundamental justice "
        "demand a de novo assessment on questions of law. The accused in this case was "
        "acquitted on all counts."
    )
    metadata = LegalDocumentMetadata(source_path="/tmp/test.txt")
    return RawDocument(metadata=metadata, text=text)


def test_produces_parent_and_child_chunks(sample_doc: RawDocument) -> None:
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    chunks = chunker.chunk(sample_doc)

    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]

    assert len(parents) >= 1, "Expected at least one parent chunk"
    assert len(children) >= 1, "Expected at least one child chunk"


def test_child_chunks_reference_valid_parent(sample_doc: RawDocument) -> None:
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    chunks = chunker.chunk(sample_doc)

    parent_ids = {c.chunk_id for c in chunks if c.parent_chunk_id is None}
    for child in [c for c in chunks if c.parent_chunk_id is not None]:
        assert child.parent_chunk_id in parent_ids


def test_metadata_propagated(sample_doc: RawDocument) -> None:
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(sample_doc)
    for chunk in chunks:
        if chunk.metadata is not None:
            assert chunk.metadata.source_path == "/tmp/test.txt"


def test_chunk_ids_are_stable_across_runs(sample_doc: RawDocument) -> None:
    """Re-chunking the same document must produce identical chunk_ids."""
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    run1 = {c.chunk_id for c in chunker.chunk(sample_doc)}
    run2 = {c.chunk_id for c in chunker.chunk(sample_doc)}
    assert run1 == run2, "chunk_ids changed between runs – duplicates would be created"


def test_doc_id_is_stable_for_same_source_path() -> None:
    """Two metadata objects with the same source_path must have the same doc_id."""
    from legalrag.core.models import LegalDocumentMetadata
    m1 = LegalDocumentMetadata(source_path="/data/case.txt")
    m2 = LegalDocumentMetadata(source_path="/data/case.txt")
    assert m1.doc_id == m2.doc_id


def test_doc_id_differs_for_different_source_paths() -> None:
    from legalrag.core.models import LegalDocumentMetadata
    m1 = LegalDocumentMetadata(source_path="/data/case_a.txt")
    m2 = LegalDocumentMetadata(source_path="/data/case_b.txt")
    assert m1.doc_id != m2.doc_id


def test_doc_id_from_citation_is_path_independent() -> None:
    """doc_id_from_citation must return the same ID regardless of file location."""
    from legalrag.core.models import doc_id_from_citation
    id1 = doc_id_from_citation("2010 BCCA 220", "/old/path/file.txt")
    id2 = doc_id_from_citation("2010 BCCA 220", "/new/renamed/folder/file.txt")
    assert id1 == id2, "doc_id must not change when the file is moved or renamed"


def test_doc_id_from_citation_falls_back_to_stem() -> None:
    """Without a citation the ID must be derived from the filename stem, not full path."""
    from legalrag.core.models import doc_id_from_citation
    id1 = doc_id_from_citation(None, "/old/dir/2010 BCCA 220 Smith v Jones.txt")
    id2 = doc_id_from_citation(None, "/new/dir/2010 BCCA 220 Smith v Jones.txt")
    assert id1 == id2, "stem-based fallback ID must be stable across directory renames"


def test_doc_id_citation_differs_from_stem_fallback() -> None:
    """Citation-based and stem-based IDs for the 'same' document must be distinct."""
    from legalrag.core.models import doc_id_from_citation
    citation_id = doc_id_from_citation("2010 BCCA 220", "/data/case.txt")
    stem_id = doc_id_from_citation(None, "/data/case.txt")
    assert citation_id != stem_id
