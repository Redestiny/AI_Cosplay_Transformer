"""
InstantID Cosplay Transformation API
Main FastAPI application for real-to-anime style transfer
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import asyncio
import logging
from datetime import datetime
import os

from services.task_manager import TaskManager
from services.preprocessing import PreprocessingService
from services.instantid_pipeline import InstantIDPipeline
from services.postprocessing import PostprocessingService
from services.storage import StorageService
from models.task import Task, TaskStatus
from config.settings import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="InstantID Cosplay API",
    description="Real-to-anime style transfer using InstantID",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
settings = get_settings()
task_manager = TaskManager()
preprocessing_service = PreprocessingService()
instantid_pipeline = InstantIDPipeline()
postprocessing_service = PostprocessingService()
storage_service = StorageService()

# Request/Response models
class ConvertRequest(BaseModel):
    style_strength: float = 0.8
    id_weight: float = 0.5
    output_resolution: str = "1024x1024"
    background_removal: bool = True
    enhancement: bool = True

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    estimated_time: Optional[int] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None

# API Endpoints

@app.post("/api/v1/convert", response_model=TaskResponse)
async def convert_image(
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    style_strength: float = Form(0.8),
    id_weight: float = Form(0.5),
    output_resolution: str = Form("1024x1024"),
    background_removal: bool = Form(True),
    enhancement: bool = Form(True)
):
    """
    Submit a new image conversion task
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task object
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            parameters={
                "style_strength": style_strength,
                "id_weight": id_weight,
                "output_resolution": output_resolution,
                "background_removal": background_removal,
                "enhancement": enhancement
            }
        )
        
        # Store task
        await task_manager.create_task(task)
        
        # Persist uploads to storage for reliable background processing
        source_bytes = await source_image.read()
        style_bytes = await style_image.read()
        # Reset file pointers not needed after read; we pass bytes paths instead
        source_filename = f"source_{task_id}_{source_image.filename or 'image'}.png"
        style_filename = f"style_{task_id}_{style_image.filename or 'image'}.png"
        source_path = await storage_service.save_bytes(source_bytes, source_filename, folder="uploads")
        style_path = await storage_service.save_bytes(style_bytes, style_filename, folder="uploads")

        # Start background processing using persisted paths
        background_tasks.add_task(
            process_conversion_task,
            task_id,
            source_path,
            style_path,
            task.parameters
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Task submitted successfully"
        )
        
    except Exception as e:
        logging.error(f"Error creating conversion task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create conversion task")

@app.get("/api/v1/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a conversion task
    """
    try:
        task = await task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return StatusResponse(
            task_id=task_id,
            status=task.status.value,
            progress=task.progress,
            estimated_time=task.estimated_time,
            result_url=task.result_url,
            error_message=task.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get task status")

@app.get("/api/v1/result/{task_id}")
async def get_result(task_id: str):
    """
    Get the result image for a completed task
    """
    try:
        task = await task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Task not completed yet")
        
        if not task.result_url:
            raise HTTPException(status_code=404, detail="Result not available")
        
        # Return file response
        return FileResponse(
            path=task.result_url,
            media_type="image/png",
            filename=f"result_{task_id}.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting result: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get result")

@app.delete("/api/v1/delete/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task and its associated files
    """
    try:
        task = await task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Delete files from storage
        if task.result_url:
            await storage_service.delete_file(task.result_url)
        
        # Delete task from database
        await task_manager.delete_task(task_id)
        
        return {"message": "Task deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete task")

@app.post("/api/v1/batch")
async def batch_convert(
    background_tasks: BackgroundTasks,
    source_images: list[UploadFile] = File(...),
    style_image: UploadFile = File(...),
    style_strength: float = Form(0.8),
    id_weight: float = Form(0.5),
    output_resolution: str = Form("1024x1024")
):
    """
    Batch conversion for multiple source images
    """
    try:
        task_ids = []
        
        # Persist shared style image once
        style_bytes = await style_image.read()
        style_task_id = str(uuid.uuid4())
        style_filename = f"style_{style_task_id}_{style_image.filename or 'image'}.png"
        style_path = await storage_service.save_bytes(style_bytes, style_filename, folder="uploads")

        for source_image in source_images:
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow(),
                parameters={
                    "style_strength": style_strength,
                    "id_weight": id_weight,
                    "output_resolution": output_resolution,
                    "background_removal": True,
                    "enhancement": True
                }
            )
            
            await task_manager.create_task(task)
            # Persist each source image
            source_bytes = await source_image.read()
            source_filename = f"source_{task_id}_{source_image.filename or 'image'}.png"
            source_path = await storage_service.save_bytes(source_bytes, source_filename, folder="uploads")

            background_tasks.add_task(
                process_conversion_task,
                task_id,
                source_path,
                style_path,
                task.parameters
            )
            
            task_ids.append(task_id)
        
        return {"task_ids": task_ids, "message": f"Batch processing started for {len(task_ids)} images"}
        
    except Exception as e:
        logging.error(f"Error creating batch task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create batch task")

# Background processing function
async def process_conversion_task(
    task_id: str,
    source_path: str,
    style_path: str,
    parameters: Dict[str, Any]
):
    """
    Background task for processing image conversion
    """
    try:
        # Update task status to running
        await task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # Step 1: Preprocessing
        await task_manager.update_task_progress(task_id, 10, "Preprocessing images...")
        # Wrap saved paths into pseudo UploadFile-like objects
        class _File:
            def __init__(self, path: str):
                self.path = path
                self.filename = os.path.basename(path)
            async def read(self):
                with open(self.path, 'rb') as f:
                    return f.read()

        preprocessed_data = await preprocessing_service.process_images(
            _File(source_path), _File(style_path)
        )
        
        # Step 2: InstantID Pipeline
        await task_manager.update_task_progress(task_id, 50, "Running InstantID pipeline...")
        instantid_result = await instantid_pipeline.generate(
            preprocessed_data, parameters
        )
        
        # Step 3: Postprocessing
        await task_manager.update_task_progress(task_id, 80, "Enhancing and finalizing...")
        final_result = await postprocessing_service.enhance(
            instantid_result, parameters
        )
        
        # Step 4: Upload result
        await task_manager.update_task_progress(task_id, 90, "Uploading result...")
        result_url = await storage_service.upload_file(final_result, f"result_{task_id}.png")
        
        # Complete task
        await task_manager.update_task_status(
            task_id, TaskStatus.COMPLETED, 
            progress=100, result_url=result_url
        )
        
    except Exception as e:
        logging.error(f"Error processing task {task_id}: {str(e)}")
        await task_manager.update_task_status(
            task_id, TaskStatus.FAILED, 
            error_message=str(e)
        )

@app.get("/")
async def root():
    """
    Serve the main application page
    """
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
