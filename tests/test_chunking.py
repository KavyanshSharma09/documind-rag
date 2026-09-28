from src.ingest import PageRecord, chunk_pages


def test_chunk_pages_respects_size_and_overlap() -> None:
    pages = [
        PageRecord(
            text="a" * 1000,
            page_number=1,
            source_file="doc.txt",
        )
    ]
    chunks = chunk_pages(pages, chunk_size=400, overlap=100)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 400 for c in chunks)
    assert chunks[0].source_file == "doc.txt"
    assert chunks[0].page_number == 1


def test_chunk_pages_skips_empty_pages() -> None:
    pages = [
        PageRecord(text="", page_number=1, source_file="a.pdf"),
        PageRecord(text="hello world", page_number=2, source_file="a.pdf"),
    ]
    chunks = chunk_pages(pages, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunk_pages_invalid_overlap() -> None:
    pages = [PageRecord(text="abc", page_number=1, source_file="x.txt")]
    try:
        chunk_pages(pages, chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
