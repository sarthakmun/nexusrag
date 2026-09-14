"""
NexusRAG: Modular Late-Interaction Multi-Vector Retrieval and Contextual RAG Engine
Author: Sarthak Mun (sarthak.mun03@gmail.com)
GitHub: https://github.com/sarthakmun/nexusrag
"""

__version__ = "1.0.0"
__author__ = "Sarthak Mun"
__email__ = "sarthak.mun03@gmail.com"

from ragatouille.RAGPretrainedModel import RAGPretrainedModel
from ragatouille.RAGTrainer import RAGTrainer

__all__ = ["RAGPretrainedModel", "RAGTrainer", "__version__", "__author__"]
