from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    model_config = ConfigDict(extra="allow")


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "aggregate-server"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
