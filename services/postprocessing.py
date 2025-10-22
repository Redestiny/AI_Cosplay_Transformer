"""
Postprocessing service for image enhancement and finalization
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Dict, Any, Optional
import os
from config.settings import get_settings

class PostprocessingService:
    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        self.gfpgan_model = None
        self.esrgan_model = None
        self._load_models()
    
    def _load_models(self):
        """Load postprocessing models"""
        try:
            # Load GFPGAN for face enhancement
            self._load_gfpgan()
            
            # Load ESRGAN for super-resolution
            self._load_esrgan()
            
        except Exception as e:
            print(f"Error loading postprocessing models: {e}")
            # Continue without models for basic processing
    
    def _load_gfpgan(self):
        """Load GFPGAN model for face enhancement"""
        try:
            # This would load the actual GFPGAN model
            # For now, we'll use a placeholder
            print("GFPGAN model placeholder - would load GFPGAN for face enhancement")
            self.gfpgan_model = None
            
        except Exception as e:
            print(f"Error loading GFPGAN: {e}")
            self.gfpgan_model = None
    
    def _load_esrgan(self):
        """Load ESRGAN model for super-resolution"""
        try:
            # This would load the actual ESRGAN model
            # For now, we'll use a placeholder
            print("ESRGAN model placeholder - would load ESRGAN for super-resolution")
            self.esrgan_model = None
            
        except Exception as e:
            print(f"Error loading ESRGAN: {e}")
            self.esrgan_model = None
    
    async def enhance(self, image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Enhance the generated image with postprocessing
        """
        try:
            enhanced_image = image.copy()
            
            # Apply face enhancement if enabled
            if parameters.get("enhancement", True):
                enhanced_image = await self._enhance_faces(enhanced_image)
            
            # Apply super-resolution if needed
            if parameters.get("super_resolution", False):
                enhanced_image = await self._apply_super_resolution(enhanced_image)
            
            # Apply background removal if requested
            if parameters.get("background_removal", True):
                enhanced_image = await self._remove_background(enhanced_image)
            
            # Apply final adjustments
            enhanced_image = await self._apply_final_adjustments(enhanced_image, parameters)
            
            return enhanced_image
            
        except Exception as e:
            raise Exception(f"Error in postprocessing: {str(e)}")
    
    async def _enhance_faces(self, image: np.ndarray) -> np.ndarray:
        """Enhance faces using GFPGAN"""
        try:
            if not self.gfpgan_model:
                # Fallback to basic face enhancement
                return await self._basic_face_enhancement(image)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply GFPGAN enhancement
            # This would use the actual GFPGAN model
            # For now, return the original image
            return image
            
        except Exception as e:
            print(f"Error enhancing faces: {e}")
            return image
    
    async def _basic_face_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Basic face enhancement without GFPGAN"""
        try:
            # Apply basic sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original
            enhanced = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in basic face enhancement: {e}")
            return image
    
    async def _apply_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """Apply super-resolution using ESRGAN"""
        try:
            if not self.esrgan_model:
                # Fallback to basic upscaling
                return await self._basic_upscaling(image)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply ESRGAN super-resolution
            # This would use the actual ESRGAN model
            # For now, return the original image
            return image
            
        except Exception as e:
            print(f"Error applying super-resolution: {e}")
            return image
    
    async def _basic_upscaling(self, image: np.ndarray) -> np.ndarray:
        """Basic upscaling without ESRGAN"""
        try:
            # Use Lanczos interpolation for better quality
            h, w = image.shape[:2]
            upscaled = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
            return upscaled
            
        except Exception as e:
            print(f"Error in basic upscaling: {e}")
            return image
    
    async def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove or replace background"""
        try:
            # Use basic background removal
            # This is a simplified version - in production, you'd use a proper segmentation model
            
            # Convert to HSV for better color-based segmentation
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Create mask for background (assuming light background)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Invert mask to get foreground
            mask = cv2.bitwise_not(mask)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Create transparent background
            result = image.copy()
            result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
            result[:, :, 3] = mask
            
            return result
            
        except Exception as e:
            print(f"Error removing background: {e}")
            return image
    
    async def _apply_final_adjustments(self, image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply final adjustments to the image"""
        try:
            adjusted_image = image.copy()
            
            # Adjust brightness and contrast
            brightness = parameters.get("brightness", 0)
            contrast = parameters.get("contrast", 1.0)
            
            if brightness != 0 or contrast != 1.0:
                adjusted_image = self._adjust_brightness_contrast(adjusted_image, brightness, contrast)
            
            # Apply color correction
            saturation = parameters.get("saturation", 1.0)
            if saturation != 1.0:
                adjusted_image = self._adjust_saturation(adjusted_image, saturation)
            
            # Apply sharpening
            if parameters.get("sharpen", False):
                adjusted_image = self._apply_sharpening(adjusted_image)
            
            return adjusted_image
            
        except Exception as e:
            print(f"Error applying final adjustments: {e}")
            return image
    
    def _adjust_brightness_contrast(self, image: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
        """Adjust brightness and contrast"""
        try:
            # Apply contrast and brightness
            adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
            return adjusted
            
        except Exception as e:
            print(f"Error adjusting brightness/contrast: {e}")
            return image
    
    def _adjust_saturation(self, image: np.ndarray, saturation: float) -> np.ndarray:
        """Adjust image saturation"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Adjust saturation
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Convert back to RGB
            adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return adjusted
            
        except Exception as e:
            print(f"Error adjusting saturation: {e}")
            return image
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        try:
            # Create sharpening kernel
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            return result
            
        except Exception as e:
            print(f"Error applying sharpening: {e}")
            return image
    
    def resize_to_target(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image to target size"""
        try:
            target_w, target_h = target_size
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            return resized
            
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
    
    def add_watermark(self, image: np.ndarray, watermark_text: str = "InstantID") -> np.ndarray:
        """Add watermark to image"""
        try:
            # Create a copy to avoid modifying original
            watermarked = image.copy()
            
            # Get image dimensions
            h, w = watermarked.shape[:2]
            
            # Set font properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)  # White
            thickness = 1
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(watermark_text, font, font_scale, thickness)
            
            # Position watermark in bottom-right corner
            x = w - text_w - 10
            y = h - 10
            
            # Add text with background
            cv2.rectangle(watermarked, (x-5, y-text_h-5), (x+text_w+5, y+5), (0, 0, 0), -1)
            cv2.putText(watermarked, watermark_text, (x, y), font, font_scale, color, thickness)
            
            return watermarked
            
        except Exception as e:
            print(f"Error adding watermark: {e}")
            return image
