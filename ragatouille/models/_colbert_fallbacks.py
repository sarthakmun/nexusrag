import sys
from pathlib import Path

try:
    from colbert import Indexer, IndexUpdater, Searcher, Trainer
    from colbert.indexing.collection_indexer import CollectionIndexer
    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.modeling.colbert import ColBERT as ColBERTModel
    HAS_COLBERT = True
except (ImportError, ModuleNotFoundError):
    HAS_COLBERT = False

    class ColBERTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.bsize = kwargs.get("bsize", 32)
            self.max_tokens = kwargs.get("max_tokens", 510)
            self.nbits = kwargs.get("nbits", 2)
            self.kmeans_niters = kwargs.get("kmeans_niters", 4)
            self.similarity = kwargs.get("similarity", "cosine")

        @classmethod
        def load_from_checkpoint(cls, path):
            return cls()

        def __getattr__(self, name):
            return None

    class Run:
        @classmethod
        def init(cls):
            pass

        @classmethod
        def context(cls, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()

    class RunConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Trainer:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, *args, **kwargs):
            pass

    class Checkpoint:
        def __init__(self, *args, **kwargs):
            pass

        def docFromText(self, *args, **kwargs):
            return []

        def queryFromText(self, *args, **kwargs):
            return []

    class Indexer:
        def __init__(self, *args, **kwargs):
            pass

        def index(self, *args, **kwargs):
            pass

    class IndexUpdater:
        def __init__(self, *args, **kwargs):
            pass

    class Searcher:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return []

        def search_all(self, *args, **kwargs):
            return []

    class CollectionIndexer:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def _train_kmeans(*args, **kwargs):
            pass

    class ColBERTModel:
        def __init__(self, *args, **kwargs):
            pass
