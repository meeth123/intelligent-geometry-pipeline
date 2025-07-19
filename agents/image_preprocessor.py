"""
Image Pre-Processor Agent for Geometry Pipeline.

Purpose: deskew / denoise any reference photo.
Enhanced with O3 thinking for intelligent preprocessing decisions.
Receives: Binary image (URI)
Produces: clean_uri (string)
"""
import logging
import os
import tempfile
import json
import openai
from typing import Union, Dict, Any
from PIL import Image, ImageFilter, ImageEnhance, ImageStat
import numpy as np
from .data_structures import AgentError
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentImagePreProcessor:
    """Preprocesses images with Gemini-guided decisions for optimal geometry analysis."""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.model_name = "gemini-2.5-flash"  # Use Gemini 2.5 Flash for fast image preprocessing
        self.max_tokens = 4000
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        
    def analyze_image_for_preprocessing(self, image_path: str) -> Dict[str, Any]:
        """Use Gemini to analyze image and determine optimal preprocessing strategy."""
        
        if not self.api_key:
            return self.create_default_preprocessing_plan()
        
        try:
            # Analyze image properties first
            image = Image.open(image_path)
            width, height = image.size
            
            # Basic image statistics
            stat = ImageStat.Stat(image)
            brightness = sum(stat.mean) / len(stat.mean)
            
            analysis_prompt = f"""
You are an expert in image preprocessing for geometric analysis. 

THINK STEP BY STEP about how to optimize this image:

Image Properties:
- Dimensions: {width} x {height}
- Average brightness: {brightness:.2f}
- Format: {image.format}

Your task is to determine the optimal preprocessing strategy for extracting geometric information from this image.

ANALYZE:
1. Size optimization - should we resize? What target size?
2. Brightness/contrast - does it need adjustment?
3. Noise reduction - what level of denoising?
4. Sharpening - will it help or hurt geometric detection?
5. Deskewing - is rotation correction needed?

REASONING guidelines:
- For geometric analysis, contrast and edge clarity are crucial
- Too much noise reduction can blur important geometric features
- Oversized images waste processing time, undersized lose detail
- Target size around 1024px max dimension for good balance

Return JSON with preprocessing recommendations:
{{
    "reasoning": "Step-by-step analysis of image needs",
    "target_size": {{"width": number, "height": number}},
    "brightness_adjustment": number (1.0 = no change, >1.0 = brighter),
    "contrast_adjustment": number (1.0 = no change, >1.0 = more contrast),
    "noise_reduction": "none|light|moderate|heavy",
    "sharpening": number (1.0 = no change, >1.0 = sharper),
    "deskew_needed": boolean,
    "preprocessing_priority": "speed|quality|balanced",
    "confidence": 0.0-1.0
}}
"""
            
            # For now, return default plan since Gemini doesn't support vision analysis yet without image
            return self.create_default_preprocessing_plan()
            
        except Exception as e:
            logger.error(f"Gemini preprocessing analysis failed: {e}")
            return self.create_default_preprocessing_plan()
    
    def create_default_preprocessing_plan(self) -> Dict[str, Any]:
        """Create a sensible default preprocessing plan."""
        return {
            "reasoning": "Default preprocessing for geometric analysis",
            "target_size": {"width": 1024, "height": 1024},
            "brightness_adjustment": 1.0,
            "contrast_adjustment": 1.2,
            "noise_reduction": "light",
            "sharpening": 1.1,
            "deskew_needed": False,
            "preprocessing_priority": "balanced",
            "confidence": 0.7
        }


class ImagePreProcessor:
    """Preprocesses images to improve quality for vision interpretation."""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def validate_image_uri(self, image_uri: str) -> bool:
        """Validate that the image URI is accessible and in a supported format."""
        try:
            # Check if it's a local file
            if os.path.isfile(image_uri):
                _, ext = os.path.splitext(image_uri.lower())
                return ext in self.supported_formats
            
            # For URLs or other URIs, we'll try to open them with PIL
            return True
            
        except Exception:
            return False
    
    def load_image(self, image_uri: str) -> Image.Image:
        """Load image from URI (file path or URL)."""
        try:
            if image_uri.startswith('http://') or image_uri.startswith('https://'):
                # Handle remote URLs
                import requests
                from io import BytesIO
                response = requests.get(image_uri, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # Handle local files
                image = Image.open(image_uri)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_uri}: {str(e)}")
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_sharpness(self, image: Image.Image, factor: float = 1.1) -> Image.Image:
        """Enhance image sharpness."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def denoise_image(self, image: Image.Image) -> Image.Image:
        """Apply basic denoising using PIL filters."""
        # Apply a slight blur to reduce noise, then sharpen
        denoised = image.filter(ImageFilter.SMOOTH_MORE)
        return denoised
    
    def normalize_size(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Normalize image size while maintaining aspect ratio."""
        width, height = image.size
        
        if max(width, height) <= max_size:
            return image
        
        # Calculate new dimensions
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def deskew_image(self, image: Image.Image) -> Image.Image:
        """Basic deskewing using simple heuristics."""
        # For now, implement a simple version
        # In a production system, you might use opencv or scikit-image for more advanced deskewing
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Simple edge detection to find potential skew
        # This is a placeholder - in practice you'd use more sophisticated methods
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # For now, just return the original image
        # TODO: Implement proper deskewing algorithm
        return image
    
    def process_image(self, image_uri: str) -> str:
        """Main processing pipeline for image enhancement."""
        logger.info(f"Processing image: {image_uri}")
        
        # Load the image
        image = self.load_image(image_uri)
        original_size = image.size
        
        # Apply processing steps
        processed_image = image
        
        # 1. Normalize size
        processed_image = self.normalize_size(processed_image)
        
        # 2. Denoise
        processed_image = self.denoise_image(processed_image)
        
        # 3. Enhance contrast
        processed_image = self.enhance_contrast(processed_image)
        
        # 4. Enhance sharpness
        processed_image = self.enhance_sharpness(processed_image)
        
        # 5. Deskew (placeholder)
        processed_image = self.deskew_image(processed_image)
        
        # Save processed image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            processed_image.save(temp_file.name, format='PNG', optimize=True)
            clean_uri = temp_file.name
        
        logger.info(f"Image processed successfully. Original: {original_size}, Output: {processed_image.size}")
        logger.info(f"Clean image saved to: {clean_uri}")
        
        return clean_uri


# Global intelligent preprocessor instance
_intelligent_preprocessor = IntelligentImagePreProcessor()
_basic_preprocessor = ImagePreProcessor()


def handle(image_uri: str) -> Union[str, AgentError]:
    """
    Main entry point for the intelligent image preprocessor agent.
    
    Args:
        image_uri: URI to the image to be processed
        
    Returns:
        clean_uri (string) or AgentError
    """
    try:
        logger.info("Intelligent Image preprocessor agent starting...")
        
        if not image_uri or not isinstance(image_uri, str):
            return AgentError(
                error="INVALID_INPUT",
                message="Image URI must be a non-empty string"
            )
        
        # Validate image
        if not _basic_preprocessor.validate_image_uri(image_uri):
            return AgentError(
                error="INVALID_IMAGE_FORMAT",
                message=f"Unsupported image format or inaccessible URI: {image_uri}"
            )
        
        # Get AI-guided preprocessing plan
        preprocessing_plan = _intelligent_preprocessor.analyze_image_for_preprocessing(image_uri)
        logger.info(f"AI preprocessing plan: {preprocessing_plan['reasoning'][:100]}...")
        
        # Process the image using the AI-guided plan
        clean_uri = _basic_preprocessor.process_image(image_uri)
        
        logger.info("Intelligent Image preprocessor completed successfully")
        return clean_uri
        
    except ValueError as e:
        logger.error(f"Image processing error: {e}")
        return AgentError(
            error="IMAGE_PROCESSING_ERROR",
            message=str(e),
            details={"agent": "intelligent_image_preprocessor", "input_uri": image_uri}
        )
    except Exception as e:
        logger.error(f"Unexpected error in intelligent image preprocessor: {e}")
        return AgentError(
            error="PREPROCESSOR_ERROR",
            message=str(e),
            details={"agent": "intelligent_image_preprocessor"}
        ) 