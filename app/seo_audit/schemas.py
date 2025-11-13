from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class AuditSummary(BaseModel):
    total: int
    passed: int
    failed: int
    issues: int
    inprogress: int

class AuditBase(BaseModel):
    id: int
    site: Optional[str]
    input_url: Optional[str]
    canonical_url: Optional[str]
    date: str
    status: str

class AuditDetail(AuditBase):
    details: Dict[str, Any]
    recommendations: List[Any]

class AuditCreate(BaseModel):
    url: str 