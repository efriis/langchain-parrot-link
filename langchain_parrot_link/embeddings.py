from typing import List

from langchain_core.embeddings import Embeddings


class ParrotLinkEmbeddings(Embeddings):
    """ParrotLink embedding model integration.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain-parrot-link`` and set environment variable ``PARROTLINK_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-parrot-link
            export PARROTLINK_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of ParrotLink model to use.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_parrot_link import ParrotLinkEmbeddings

            embed = ParrotLinkEmbeddings(
                model="...",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Embed multiple text:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await embed.aembed_query(input_text)

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            # TODO: Example output.

    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [[1.0, 2.5, 3.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
