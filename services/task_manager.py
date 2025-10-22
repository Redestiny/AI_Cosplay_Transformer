"""
Task management service using Redis for queue and storage
"""

import json
import asyncio
from typing import Optional, List
from datetime import datetime
import redis.asyncio as redis
from models.task import Task, TaskStatus
from config.settings import get_settings

class TaskManager:
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True
            )
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory storage for development
            self.redis_client = None
            self._memory_storage = {}
    
    async def create_task(self, task: Task) -> bool:
        """Create a new task"""
        try:
            task_data = task.dict()
            task_data['created_at'] = task.created_at.isoformat()
            if task.updated_at:
                task_data['updated_at'] = task.updated_at.isoformat()
            if task.completed_at:
                task_data['completed_at'] = task.completed_at.isoformat()
            
            if self.redis_client:
                await self.redis_client.hset(
                    f"task:{task.task_id}",
                    mapping=task_data
                )
                await self.redis_client.lpush("task_queue", task.task_id)
            else:
                self._memory_storage[task.task_id] = task_data
            
            return True
        except Exception as e:
            print(f"Error creating task: {e}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        try:
            if self.redis_client:
                task_data = await self.redis_client.hgetall(f"task:{task_id}")
                if not task_data:
                    return None
            else:
                task_data = self._memory_storage.get(task_id)
                if not task_data:
                    return None
            
            # Convert string values back to appropriate types
            if 'created_at' in task_data:
                task_data['created_at'] = datetime.fromisoformat(task_data['created_at'])
            if 'updated_at' in task_data and task_data['updated_at']:
                task_data['updated_at'] = datetime.fromisoformat(task_data['updated_at'])
            if 'completed_at' in task_data and task_data['completed_at']:
                task_data['completed_at'] = datetime.fromisoformat(task_data['completed_at'])
            
            # Convert progress to int
            if 'progress' in task_data:
                task_data['progress'] = int(task_data['progress'])
            if 'estimated_time' in task_data and task_data['estimated_time']:
                task_data['estimated_time'] = int(task_data['estimated_time'])
            
            return Task(**task_data)
        except Exception as e:
            print(f"Error getting task: {e}")
            return None
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        progress: Optional[int] = None,
        result_url: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status and related fields"""
        try:
            update_data = {
                'status': status.value,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if progress is not None:
                update_data['progress'] = progress
            
            if result_url:
                update_data['result_url'] = result_url
            
            if error_message:
                update_data['error_message'] = error_message
            
            if status == TaskStatus.COMPLETED:
                update_data['completed_at'] = datetime.utcnow().isoformat()
            
            if self.redis_client:
                await self.redis_client.hset(f"task:{task_id}", mapping=update_data)
            else:
                if task_id in self._memory_storage:
                    self._memory_storage[task_id].update(update_data)
            
            return True
        except Exception as e:
            print(f"Error updating task status: {e}")
            return False
    
    async def update_task_progress(self, task_id: str, progress: int, message: str = "") -> bool:
        """Update task progress"""
        try:
            update_data = {
                'progress': progress,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if self.redis_client:
                await self.redis_client.hset(f"task:{task_id}", mapping=update_data)
            else:
                if task_id in self._memory_storage:
                    self._memory_storage[task_id].update(update_data)
            
            return True
        except Exception as e:
            print(f"Error updating task progress: {e}")
            return False
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        try:
            if self.redis_client:
                await self.redis_client.delete(f"task:{task_id}")
                await self.redis_client.lrem("task_queue", 0, task_id)
            else:
                if task_id in self._memory_storage:
                    del self._memory_storage[task_id]
            
            return True
        except Exception as e:
            print(f"Error deleting task: {e}")
            return False
    
    async def get_pending_tasks(self) -> List[str]:
        """Get list of pending task IDs"""
        try:
            if self.redis_client:
                return await self.redis_client.lrange("task_queue", 0, -1)
            else:
                return [task_id for task_id, data in self._memory_storage.items() 
                       if data.get('status') == TaskStatus.PENDING.value]
        except Exception as e:
            print(f"Error getting pending tasks: {e}")
            return []
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
