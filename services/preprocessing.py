"""
Preprocessing service for face/body detection and alignment
"""

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import Dict, Any, Tuple
import torch
import torchvision.transforms as transforms
from fastapi import UploadFile
import os
import io

class PreprocessingService:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # Initialize models
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
    
    async def process_images(self, source_image: UploadFile, style_image: UploadFile) -> Dict[str, Any]:
        """
        Process source and style images for InstantID pipeline
        """
        try:
            # Load images
            source_img = await self._load_image(source_image)
            style_img = await self._load_image(style_image)
            
            # Detect and extract face from source image
            source_face_data = await self._extract_face_data(source_img)
            
            # Detect and extract face from style image
            style_face_data = await self._extract_face_data(style_img)
            
            # Extract pose information
            source_pose = await self._extract_pose(source_img)
            style_pose = await self._extract_pose(style_img)
            
            # Extract body segmentation
            source_segmentation = await self._extract_segmentation(source_img)
            style_segmentation = await self._extract_segmentation(style_img)
            
            return {
                "source_image": source_img,
                "style_image": style_img,
                "source_face": source_face_data,
                "style_face": style_face_data,
                "source_pose": source_pose,
                "style_pose": style_pose,
                "source_segmentation": source_segmentation,
                "style_segmentation": style_segmentation
            }
            
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")
    
    async def _load_image(self, upload_file: UploadFile) -> np.ndarray:
        """Load image from UploadFile. Can also accept a pseudo UploadFile with .read()/.filename."""
        try:
            content = await upload_file.read()
            image = Image.open(io.BytesIO(content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    async def _extract_face_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract face landmarks and features"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_results = self.face_detection.process(rgb_image)
            face_detections = []
            
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]
                    face_detections.append({
                        'bbox': bbox,
                        'confidence': confidence
                    })
            
            # Face mesh for landmarks
            face_mesh_results = self.face_mesh.process(rgb_image)
            landmarks = []
            
            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
            
            return {
                'detections': face_detections,
                'landmarks': landmarks,
                'has_face': len(face_detections) > 0
            }
            
        except Exception as e:
            raise Exception(f"Error extracting face data: {str(e)}")
    
    async def _extract_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract pose information"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_image)
            
            landmarks = []
            if pose_results.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z, lm.visibility] 
                           for lm in pose_results.pose_landmarks.landmark]
            
            return {
                'landmarks': landmarks,
                'has_pose': len(landmarks) > 0
            }
            
        except Exception as e:
            raise Exception(f"Error extracting pose: {str(e)}")
    
    async def _extract_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Extract body segmentation mask"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segmentation_results = self.selfie_segmentation.process(rgb_image)
            
            if segmentation_results.segmentation_mask is not None:
                return segmentation_results.segmentation_mask
            else:
                # Return empty mask if no segmentation
                return np.zeros(image.shape[:2], dtype=np.uint8)
                
        except Exception as e:
            raise Exception(f"Error extracting segmentation: {str(e)}")
    
    def align_face(self, image: np.ndarray, landmarks: list) -> np.ndarray:
        """Align face based on landmarks"""
        try:
            if not landmarks:
                return image
            
            # Get eye landmarks (assuming standard face mesh)
            left_eye = landmarks[33]  # Left eye center
            right_eye = landmarks[263]  # Right eye center
            
            # Calculate rotation angle
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate image to align eyes horizontally
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))
            
            return aligned_image
            
        except Exception as e:
            print(f"Error aligning face: {e}")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create canvas with target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
            
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
