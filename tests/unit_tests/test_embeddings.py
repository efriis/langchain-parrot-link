from typing import Tuple, Type

from langchain_parrot_link.embeddings import ParrotLinkEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[ParrotLinkEmbeddings]:
        return ParrotLinkEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {}
