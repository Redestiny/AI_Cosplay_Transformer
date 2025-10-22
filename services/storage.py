"""
Storage service for file management and CDN integration
"""

import os
import uuid
from typing import Optional, BinaryIO
import boto3
from botocore.exceptions import ClientError
import numpy as np
from PIL import Image
import io
from config.settings import get_settings

class StorageService:
    def __init__(self):
        self.settings = get_settings()
        self.storage_type = self.settings.storage_type
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage backend"""
        try:
            if self.storage_type == "s3":
                self._initialize_s3()
            elif self.storage_type == "local":
                self._initialize_local()
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")
                
        except Exception as e:
            print(f"Error initializing storage: {e}")
            # Fallback to local storage
            self.storage_type = "local"
            self._initialize_local()
    
    def _initialize_s3(self):
        """Initialize S3 storage"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                region_name=self.settings.s3_region
            )
            self.bucket_name = self.settings.s3_bucket
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
        except Exception as e:
            print(f"Error initializing S3: {e}")
            raise
    
    def _initialize_local(self):
        """Initialize local storage"""
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.settings.storage_path, exist_ok=True)
            
            # Create subdirectories
            os.makedirs(os.path.join(self.settings.storage_path, "uploads"), exist_ok=True)
            os.makedirs(os.path.join(self.settings.storage_path, "results"), exist_ok=True)
            os.makedirs(os.path.join(self.settings.storage_path, "temp"), exist_ok=True)
            
        except Exception as e:
            print(f"Error initializing local storage: {e}")
            raise
    
    async def upload_file(self, image: np.ndarray, filename: str, folder: str = "results") -> str:
        """
        Upload image file to storage
        Returns the URL or path to the uploaded file
        """
        try:
            # Generate unique filename if not provided
            if not filename:
                filename = f"{uuid.uuid4()}.png"
            
            # Ensure filename has extension
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
            
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                pil_image = Image.fromarray(image, 'RGBA')
            else:  # RGB
                pil_image = Image.fromarray(image, 'RGB')
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG', quality=95)
            img_data = img_buffer.getvalue()
            
            if self.storage_type == "s3":
                return await self._upload_to_s3(img_data, filename, folder)
            else:
                return await self._upload_to_local(img_data, filename, folder)
                
        except Exception as e:
            raise Exception(f"Error uploading file: {str(e)}")
    
    async def _upload_to_s3(self, data: bytes, filename: str, folder: str) -> str:
        """Upload file to S3"""
        try:
            key = f"{folder}/{filename}"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType='image/png'
            )
            
            # Return public URL
            url = f"https://{self.bucket_name}.s3.{self.settings.s3_region}.amazonaws.com/{key}"
            return url
            
        except ClientError as e:
            raise Exception(f"S3 upload error: {str(e)}")
    
    async def _upload_to_local(self, data: bytes, filename: str, folder: str) -> str:
        """Upload file to local storage"""
        try:
            # Create folder path
            folder_path = os.path.join(self.settings.storage_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Save file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Return file path
            return file_path
            
        except Exception as e:
            raise Exception(f"Local upload error: {str(e)}")

    async def save_bytes(self, data: bytes, filename: str, folder: str = "uploads") -> str:
        """Save raw bytes to storage (typically for uploads)."""
        try:
            if not filename:
                filename = f"{uuid.uuid4()}.bin"
            if self.storage_type == "s3":
                return await self._upload_to_s3(data, filename, folder)
            return await self._upload_to_local(data, filename, folder)
        except Exception as e:
            raise Exception(f"Error saving bytes: {str(e)}")
    
    async def download_file(self, file_path: str) -> bytes:
        """Download file from storage"""
        try:
            if self.storage_type == "s3":
                return await self._download_from_s3(file_path)
            else:
                return await self._download_from_local(file_path)
                
        except Exception as e:
            raise Exception(f"Error downloading file: {str(e)}")
    
    async def _download_from_s3(self, key: str) -> bytes:
        """Download file from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
            
        except ClientError as e:
            raise Exception(f"S3 download error: {str(e)}")
    
    async def _download_from_local(self, file_path: str) -> bytes:
        """Download file from local storage"""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
                
        except Exception as e:
            raise Exception(f"Local download error: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage"""
        try:
            if self.storage_type == "s3":
                return await self._delete_from_s3(file_path)
            else:
                return await self._delete_from_local(file_path)
                
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    async def _delete_from_s3(self, key: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
            
        except ClientError as e:
            print(f"S3 delete error: {str(e)}")
            return False
    
    async def _delete_from_local(self, file_path: str) -> bool:
        """Delete file from local storage"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
            
        except Exception as e:
            print(f"Local delete error: {str(e)}")
            return False
    
    async def get_file_url(self, file_path: str) -> str:
        """Get public URL for file"""
        try:
            if self.storage_type == "s3":
                # Generate presigned URL for S3
                return self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': file_path},
                    ExpiresIn=3600  # 1 hour
                )
            else:
                # For local storage, return file path
                return file_path
                
        except Exception as e:
            print(f"Error getting file URL: {e}")
            return file_path
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up old files from storage"""
        try:
            cleaned_count = 0
            
            if self.storage_type == "s3":
                cleaned_count = await self._cleanup_s3_files(max_age_hours)
            else:
                cleaned_count = await self._cleanup_local_files(max_age_hours)
            
            return cleaned_count
            
        except Exception as e:
            print(f"Error cleaning up files: {e}")
            return 0
    
    async def _cleanup_s3_files(self, max_age_hours: int) -> int:
        """Clean up old files from S3"""
        try:
            import datetime
            
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            # List objects in bucket
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_time:
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            print(f"Error cleaning up S3 files: {e}")
            return 0
    
    async def _cleanup_local_files(self, max_age_hours: int) -> int:
        """Clean up old files from local storage"""
        try:
            import time
            
            cleaned_count = 0
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Walk through storage directory
            for root, dirs, files in os.walk(self.settings.storage_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            print(f"Error cleaning up local files: {e}")
            return 0
    
    def get_storage_info(self) -> dict:
        """Get storage information"""
        try:
            info = {
                "type": self.storage_type,
                "path": self.settings.storage_path if self.storage_type == "local" else None,
                "bucket": self.bucket_name if self.storage_type == "s3" else None
            }
            return info
            
        except Exception as e:
            print(f"Error getting storage info: {e}")
            return {"type": "unknown", "error": str(e)}
