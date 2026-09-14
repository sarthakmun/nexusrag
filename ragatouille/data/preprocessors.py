try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
except (ImportError, ModuleNotFoundError):
    try:
        from llama_index import Document
        from llama_index.text_splitter import SentenceSplitter
    except (ImportError, ModuleNotFoundError):
        try:
            from llama_index.core.schema import Document
            from llama_index.core.text_splitter import SentenceSplitter
        except (ImportError, ModuleNotFoundError):
            class Document:
                def __init__(self, text="", **kwargs):
                    self.text = text
            class SentenceSplitter:
                def __init__(self, chunk_size=256, chunk_overlap=64, **kwargs):
                    self.chunk_size = chunk_size
                    self.chunk_overlap = chunk_overlap
                def __call__(self, docs):
                    class Node:
                        def __init__(self, text):
                            self.text = text
                    nodes = []
                    for d in docs:
                        text = getattr(d, "text", str(d))
                        for i in range(0, max(1, len(text)), self.chunk_size):
                            nodes.append(Node(text[i:i + self.chunk_size]))
                    return nodes


def llama_index_sentence_splitter(
    documents: list[str], document_ids: list[str], chunk_size=256
):
    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    chunks = []
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [[Document(text=doc)] for doc in documents]
    for doc_id, doc in zip(document_ids, docs):
        chunks += [
            {"document_id": doc_id, "content": node.text} for node in node_parser(doc)
        ]
    return chunks
