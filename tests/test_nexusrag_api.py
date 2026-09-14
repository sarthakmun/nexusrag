import pytest
import nexusrag
from nexusrag import RAGPretrainedModel, RAGTrainer


def test_nexusrag_version_and_metadata():
    assert nexusrag.__version__ == "1.0.0"
    assert nexusrag.__author__ == "Sarthak Mun"
    assert "sarthak.mun03@gmail.com" in nexusrag.__email__


def test_nexusrag_class_exports():
    assert RAGPretrainedModel is not None
    assert RAGTrainer is not None
    assert hasattr(RAGPretrainedModel, "from_pretrained")
    assert hasattr(RAGPretrainedModel, "from_index")
    assert hasattr(RAGTrainer, "train")
