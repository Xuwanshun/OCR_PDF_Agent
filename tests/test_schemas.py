from ocr_agent.schemas import DocumentJSON, PageContent, TextBlock


def test_document_json_serialization():
    doc = DocumentJSON(
        doc_hash="h",
        source_path="x.pdf",
        pages=[
            PageContent(page_number=1, width=100, height=100, blocks=[TextBlock(text="hi", bbox=[0, 0, 1, 1])]),
        ],
    )
    dumped = doc.model_dump()
    assert dumped["doc_hash"] == "h"
    assert dumped["pages"][0]["blocks"][0]["text"] == "hi"

