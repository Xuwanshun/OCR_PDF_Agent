from ocr_agent.chunking import chunk_text


def test_chunk_text_basic():
    text = "a" * 10000
    chunks = chunk_text(text, target_chars=4800, max_chars=8000)
    assert len(chunks) >= 2
    assert all(chunks)

