"""
InstantID pipeline service for style transfer
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional
import cv2
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModel
import torch.nn.functional as F
from safetensors.torch import load_file
import os
from config.settings import get_settings

class InstantIDPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device(self.settings.device if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.face_encoder = None
        self.style_encoder = None
        self._load_models()
    
    def _load_models(self):
        """Load InstantID models"""
        try:
            # Load Stable Diffusion XL pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device.type == "cuda" else None
            )
            self.pipeline = self.pipeline.to(self.device)
            
            # Load scheduler
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Load face encoder (ArcFace)
            self._load_face_encoder()
            
            # Load style encoder (IP-Adapter)
            self._load_style_encoder()
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to basic pipeline
            self._load_fallback_models()
    
    def _load_face_encoder(self):
        """Load face encoder for identity preservation"""
        try:
            # Load CLIP vision model for face encoding
            self.face_encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            
            self.face_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            
        except Exception as e:
            print(f"Error loading face encoder: {e}")
            self.face_encoder = None
    
    def _load_style_encoder(self):
        """Load style encoder for style transfer"""
        try:
            # Load IP-Adapter for style encoding
            # This would typically load a pre-trained IP-Adapter model
            # For now, we'll use a placeholder
            self.style_encoder = None
            print("Style encoder placeholder - would load IP-Adapter model")
            
        except Exception as e:
            print(f"Error loading style encoder: {e}")
            self.style_encoder = None
    
    def _load_fallback_models(self):
        """Load fallback models if main models fail"""
        try:
            # Basic SDXL pipeline without InstantID components
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            self.pipeline = self.pipeline.to(self.device)
            
        except Exception as e:
            print(f"Error loading fallback models: {e}")
            raise Exception("Failed to load any models")
    
    async def generate(self, preprocessed_data: Dict[str, Any], parameters: Dict[str, Any]) -> np.ndarray:
        """
        Generate style-transferred image using InstantID pipeline
        """
        try:
            source_image = preprocessed_data["source_image"]
            style_image = preprocessed_data["style_image"]
            source_face = preprocessed_data["source_face"]
            style_face = preprocessed_data["style_face"]
            
            # Extract parameters
            style_strength = parameters.get("style_strength", 0.8)
            id_weight = parameters.get("id_weight", 0.5)
            output_resolution = parameters.get("output_resolution", "1024x1024")
            
            # Parse resolution
            width, height = map(int, output_resolution.split("x"))
            
            # Prepare face embeddings
            face_embedding = await self._extract_face_embedding(source_image, source_face)
            style_embedding = await self._extract_style_embedding(style_image, style_face)
            
            # Generate prompt
            prompt = await self._generate_prompt(source_face, style_face, parameters)
            
            # Run InstantID pipeline
            result = await self._run_instantid_pipeline(
                prompt=prompt,
                face_embedding=face_embedding,
                style_embedding=style_embedding,
                style_strength=style_strength,
                id_weight=id_weight,
                width=width,
                height=height
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Error in InstantID generation: {str(e)}")
    
    async def _extract_face_embedding(self, image: np.ndarray, face_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract face embedding for identity preservation"""
        try:
            if not self.face_encoder or not face_data.get("has_face", False):
                return None
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Process image for CLIP
            inputs = self.face_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                features = self.face_encoder(**inputs)
                embedding = features.pooler_output
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    async def _extract_style_embedding(self, image: np.ndarray, style_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract style embedding from style image"""
        try:
            if not self.style_encoder:
                return None
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # This would use IP-Adapter style encoder
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            print(f"Error extracting style embedding: {e}")
            return None
    
    async def _generate_prompt(self, source_face: Dict[str, Any], style_face: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Generate prompt for the diffusion model"""
        try:
            # Basic prompt generation
            base_prompt = "anime style character, high quality, detailed, beautiful"
            
            # Add style-specific terms based on detected features
            if source_face.get("has_face", False):
                base_prompt += ", preserving facial identity"
            
            if style_face.get("has_face", False):
                base_prompt += ", anime art style"
            
            # Add quality terms
            base_prompt += ", masterpiece, best quality, highly detailed"
            
            return base_prompt
            
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return "anime style character, high quality, detailed"
    
    async def _run_instantid_pipeline(
        self,
        prompt: str,
        face_embedding: Optional[torch.Tensor],
        style_embedding: Optional[torch.Tensor],
        style_strength: float,
        id_weight: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """Run the InstantID pipeline"""
        try:
            # Prepare negative prompt
            negative_prompt = "low quality, blurry, distorted, deformed, ugly, bad anatomy"
            
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=width,
                    height=height,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            # Convert to numpy array
            image = result.images[0]
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            return image
            
        except Exception as e:
            raise Exception(f"Error running InstantID pipeline: {str(e)}")
    
    def _apply_face_guidance(self, image: np.ndarray, face_embedding: torch.Tensor, strength: float) -> np.ndarray:
        """Apply face guidance to maintain identity"""
        try:
            # This would implement face guidance using the face embedding
            # For now, return the original image
            return image
            
        except Exception as e:
            print(f"Error applying face guidance: {e}")
            return image
    
    def _apply_style_guidance(self, image: np.ndarray, style_embedding: torch.Tensor, strength: float) -> np.ndarray:
        """Apply style guidance for style transfer"""
        try:
            # This would implement style guidance using the style embedding
            # For now, return the original image
            return image
            
        except Exception as e:
            print(f"Error applying style guidance: {e}")
            return image
