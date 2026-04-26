from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Sample(BaseModel):
    image_path: str
    output_text: str
    source: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ModelMeta(BaseModel):
    repo: str
    revision: str
    accuracy: Optional[float] = None
    cer: Optional[float] = None
    promoted_at: Optional[datetime] = None

class InferRequest(BaseModel):
    image_b64: str
    max_tokens: int = 1024

class InferResponse(BaseModel):
    text: str
    model_revision: str
    latency_ms: float
