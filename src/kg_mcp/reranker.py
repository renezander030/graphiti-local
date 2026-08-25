"""A retrieval-only reranker that makes no additional model call."""

from graphiti_core.cross_encoder.client import CrossEncoderClient


class PassthroughReranker(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        del query
        count = max(len(passages), 1)
        return [(passage, (count - index) / count) for index, passage in enumerate(passages)]

