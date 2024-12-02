from typing import Type

from langchain_parrot_link.embeddings import ParrotLinkEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[ParrotLinkEmbeddings]:
        return ParrotLinkEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {}
