"""
Task model for managing conversion tasks
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    estimated_time: Optional[int] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True
