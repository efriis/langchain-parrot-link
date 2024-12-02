from typing import Type

from langchain_parrot_link.chat_models import ChatParrotLink
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatParrotLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
