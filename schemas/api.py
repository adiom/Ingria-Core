from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ClientBase(BaseModel):
    name: str

class ClientCreate(ClientBase):
    api_key: str

class Client(ClientBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class RequestBase(BaseModel):
    model: str
    messages: List[Dict[str, str]]

class RequestCreate(RequestBase):
    pass

class Request(RequestBase):
    id: int
    client_id: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: datetime
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    emotion: str

    class Config:
        from_attributes = True

class UsageStats(BaseModel):
    client_id: int
    total_requests: int
    total_tokens: int
    avg_response_time: float
    last_updated: datetime

    class Config:
        from_attributes = True 