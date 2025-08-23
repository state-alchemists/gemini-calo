import json
from logging import Logger
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from gemini_calo.middlewares.logging import logging_middleware


@pytest.fixture
def mock_logger():
    return MagicMock(spec=Logger)


@pytest.fixture
def client(mock_logger):
    app = FastAPI()

    @app.middleware("http")
    async def middleware(request: Request, call_next):
        return await logging_middleware(request, call_next, logger=mock_logger)

    @app.post("/regular")
    async def regular_endpoint():
        return Response(content=json.dumps({"message": "hello"}), media_type="application/json")

    @app.post("/streaming")
    async def streaming_endpoint():
        async def content():
            yield b'{"message": "hello"}'
        return StreamingResponse(content(), media_type="application/json")

    return TestClient(app)


def test_logging_middleware_regular_response(client, mock_logger):
    """
    Tests that the logging middleware logs request and response details for a regular response.
    """
    response = client.post("/regular", json={"message": "world"})
    assert response.status_code == 200
    assert mock_logger.info.call_count == 5


def test_logging_middleware_streaming_response(client, mock_logger):
    """
    Tests that the logging middleware logs request and response details for a streaming response.
    """
    response = client.post("/streaming", json={"message": "world"})
    assert response.status_code == 200
    assert mock_logger.info.call_count == 5
