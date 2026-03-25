from pathlib import Path


def test_chunk_id_format():
    doc_hash = "abc123"
    page = 2
    idx = 0
    cid = f"{doc_hash}:p{page}:text:{idx}"
    assert cid.startswith("abc123:p2:text:")

