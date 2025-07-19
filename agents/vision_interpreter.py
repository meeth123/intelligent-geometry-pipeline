"""
Vision Interpreter Agent for Geometry Pipeline.

Enhanced with Gemini Pro Vision for intelligent image analysis and geometric reasoning.
Receives: PromptBundle (read-only reference) and clean_uri (string) 
Produces: GeometrySpec fragment with vision status
"""
import logging
import json
import base64
import os
import google.generativeai as genai
from typing import Union, List, Dict, Any
from PIL import Image
from .data_structures import (
    PromptBundle, GeometrySpec, GeometryObject, GeometryConstraint,
    Status, AgentError, create_geometry_object, create_constraint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiVisionInterpreter:
    """Analyzes images using Gemini Pro Vision for geometric object detection."""
    
    def __init__(self):
        self.min_confidence_threshold = 0.5
        self.max_tokens = 8000  # Gemini optimal token count
        self.model_name = "gemini-2.5-flash"  # Gemini 2.5 Flash with vision for fast image analysis
        
        # Get API key with fallback
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # Fallback: set the key directly if not found in environment
        if not self.api_key:
            self.api_key = "AIzaSyAHCbEYuASKZkr2adn2-CanH0aF7vusnus"
            os.environ['GOOGLE_API_KEY'] = self.api_key
            logger.info("Setting Google API key directly in environment")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini vision model configured successfully")
        else:
            logger.warning("Google API key not available - vision analysis disabled")
    
    def create_vision_prompt(self, prompt_bundle: PromptBundle) -> str:
        """Create a structured prompt for the Gemini vision model with step-by-step reasoning."""
        
        base_prompt = """
You are an expert at analyzing geometric sketches and diagrams with deep mathematical reasoning.

THINK STEP BY STEP through this visual analysis:

1. OBSERVE the image carefully
   - What geometric shapes do you see?
   - How are they positioned relative to each other?
   - What are the approximate dimensions and proportions?

2. IDENTIFY spatial relationships
   - Which shapes are inside others?
   - Which shapes touch or are tangent to others?
   - Are there parallel or perpendicular relationships?
   - Do vertices of one shape lie on the boundary of another?

3. MEASURE and estimate
   - Approximate coordinates and dimensions
   - Use consistent coordinate system (0,0 at top-left)
   - Estimate confidence based on clarity and certainty

4. REASON about geometric intent
   - What geometric construction is this showing?
   - What constraints are implied by the arrangement?
   - How confident are you in each detection?

5. STRUCTURE your analysis as JSON

Return your analysis as a JSON object with this exact structure:
{
    "reasoning": "Step-by-step analysis of what you see and why",
    "objects": [
        {
            "type": "circle|triangle|square|rectangle|line|polygon|point",
            "confidence": 0.0-1.0,
            "properties": {
                "center_x": number,
                "center_y": number,
                "radius": number (for circles),
                "width": number (for rectangles),
                "height": number (for rectangles),
                "start_x": number (for lines),
                "start_y": number (for lines),
                "end_x": number (for lines),
                "end_y": number (for lines),
                "side_length": number (for squares/triangles),
                "area": number,
                "angle": number (for lines, in degrees)
            }
        }
    ],
    "constraints": [
        {
            "type": "inscribed|tangent|parallel|perpendicular|equal|distance|centered",
            "object_indices": [0, 1],
            "confidence": 0.0-1.0,
            "description": "detailed explanation of the observed relationship",
            "parameters": {
                "relationship": "specific geometric relationship observed"
            }
        }
    ],
    "overall_confidence": 0.0-1.0,
    "visual_notes": "description of what you see and any ambiguities"
}

CRITICAL REASONING GUIDELINES:
- If you see a square inside a circle with vertices touching the circle boundary, that's an inscribed square with tangent constraints
- If shapes are clearly parallel or perpendicular, include those constraints
- Be conservative with confidence scores - only high confidence for very clear relationships
- Use image coordinate system (0,0 at top-left)
- Consider the geometric intent behind the drawing

Focus on mathematical precision and clear geometric relationships.
"""
        
        # Add context from the original prompt if available
        if prompt_bundle.text:
            base_prompt += f"\n\nOriginal prompt context: \"{prompt_bundle.text}\"\n"
            base_prompt += "Use this context to help interpret ambiguous elements in the image and validate your geometric understanding."
        
        return base_prompt
    
    def call_gemini_vision_api(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Call Gemini Vision API to analyze the image."""
        
        if not self.api_key:
            raise ValueError("Google API key not configured. Set GOOGLE_API_KEY environment variable.")
        
        try:
            # Load and prepare image
            image = Image.open(image_path)
            
            # Prepare the API call
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.1,
                )
            )
            
            # Parse the response
            response_text = response.text
            logger.info(f"Gemini Vision API response: {len(response_text)} characters")
            
            # Try to parse JSON from the response
            # Sometimes the AI wraps JSON in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in AI response")
            
            parsed_json = json.loads(json_text)
            
            # Add Gemini reasoning metadata
            parsed_json["raw_gemini_reasoning"] = response_text
            parsed_json["model"] = self.model_name
            
            return parsed_json
            
        except Exception as e:
            raise ValueError(f"Gemini Vision API error: {str(e)}")
    
    def parse_gemini_response(self, gemini_response: Dict[str, Any]) -> tuple[List[GeometryObject], List[GeometryConstraint], float]:
        """Parse the Gemini response into GeometryObjects and GeometryConstraints."""
        
        objects = []
        constraints = []
        
        # Extract objects
        for obj_data in gemini_response.get("objects", []):
            obj_type = obj_data.get("type", "unknown")
            confidence = obj_data.get("confidence", 0.0)
            properties = obj_data.get("properties", {})
            
            # Only include objects with reasonable confidence
            if confidence >= 0.3:
                geometry_obj = create_geometry_object(obj_type, **properties)
                objects.append(geometry_obj)
        
        # Extract constraints
        for constraint_data in gemini_response.get("constraints", []):
            constraint_type = constraint_data.get("type", "unknown")
            object_indices = constraint_data.get("object_indices", [])
            confidence = constraint_data.get("confidence", 0.0)
            parameters = constraint_data.get("parameters", {})
            
            # Map indices to object IDs
            if len(object_indices) >= 2 and confidence >= 0.3:
                try:
                    object_ids = [objects[i].id for i in object_indices if i < len(objects)]
                    if len(object_ids) >= 2:
                        constraint = create_constraint(constraint_type, object_ids, **parameters)
                        constraints.append(constraint)
                except IndexError:
                    logger.warning(f"Invalid object indices in constraint: {object_indices}")
        
        # Get overall confidence
        overall_confidence = gemini_response.get("overall_confidence", 0.0)
        
        return objects, constraints, overall_confidence
    
    def process_image_with_ai(self, clean_uri: str, prompt_bundle: PromptBundle) -> tuple[List[GeometryObject], List[GeometryConstraint], float, str]:
        """Main AI-powered image processing pipeline with O3 reasoning capture."""
        logger.info(f"Processing image with Gemini vision: {clean_uri}")
        
        # Create vision prompt
        vision_prompt = self.create_vision_prompt(prompt_bundle)
        
        # Call AI vision API
        gemini_response = self.call_gemini_vision_api(clean_uri, vision_prompt)
        
        # Extract Gemini reasoning
        gemini_reasoning = gemini_response.get("reasoning", "No reasoning provided")
        
        # Parse response
        objects, constraints, confidence = self.parse_gemini_response(gemini_response)
        
        logger.info(f"Gemini AI detected {len(objects)} objects with overall confidence: {confidence:.2f}")
        
        return objects, constraints, confidence, gemini_reasoning


# Global interpreter instance
_ai_interpreter = GeminiVisionInterpreter()


def handle(prompt_bundle: PromptBundle, clean_uri: str) -> Union[GeometrySpec, AgentError]:
    """
    Main entry point for the AI vision interpreter agent.
    
    Args:
        prompt_bundle: PromptBundle (read-only reference)
        clean_uri: URI to the preprocessed image
        
    Returns:
        GeometrySpec fragment with vision status or AgentError
    """
    try:
        logger.info("AI Vision interpreter agent starting...")
        
        if not clean_uri or not isinstance(clean_uri, str):
            return AgentError(
                error="INVALID_INPUT",
                message="Clean URI must be a non-empty string"
            )
        
        # Process the image with AI
        objects, constraints, confidence, gemini_reasoning = _ai_interpreter.process_image_with_ai(clean_uri, prompt_bundle)
        
        # Check confidence threshold
        if confidence < _ai_interpreter.min_confidence_threshold:
            return AgentError(
                error="LOW_CONFIDENCE",
                message="needs_better_photo",
                details={
                    "confidence": confidence,
                    "threshold": _ai_interpreter.min_confidence_threshold,
                    "detected_objects": len(objects),
                    "reason": "AI vision confidence too low"
                }
            )
        
        # Create annotations with AI-specific information
        annotations = {
            'ai_vision_confidence': confidence,
            'detection_method': 'ai_vision_api',
            'ai_model': _ai_interpreter.model_name,
            'source_image': clean_uri,
            'detected_object_count': len(objects),
            'prompt_reference': prompt_bundle.prompt_id
        }
        
        # Create GeometrySpec with vision status and O3 reasoning
        geometry_spec = GeometrySpec(
            objects=objects,
            constraints=constraints,
            annotations=annotations,
            status=Status.VISION,
            confidence=confidence
        )
        
        # Add O3 vision reasoning
        geometry_spec.agent_reasoning["vision_interpreter"] = gemini_reasoning
        geometry_spec.processing_steps.append({
            "agent": "vision_interpreter",
            "step": "image_analysis", 
            "reasoning": gemini_reasoning,
            "model": _ai_interpreter.model_name,
            "confidence": confidence,
            "objects_detected": len(objects),
            "constraints_detected": len(constraints)
        })
        
        logger.info(f"AI Vision interpreter completed successfully with confidence: {confidence:.2f}")
        return geometry_spec
        
    except ValueError as e:
        if "API key" in str(e) or "Authentication" in str(e):
            return AgentError(
                error="API_KEY_MISSING",
                message="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                details={"agent": "ai_vision_interpreter", "error_detail": str(e)}
            )
        else:
            logger.error(f"AI Vision processing error: {e}")
            return AgentError(
                error="AI_VISION_ERROR",
                message=str(e),
                details={"agent": "ai_vision_interpreter", "clean_uri": clean_uri}
            )
    except Exception as e:
        logger.error(f"Unexpected error in AI vision interpreter: {e}")
        return AgentError(
            error="VISION_INTERPRETER_ERROR",
            message=str(e),
            details={"agent": "ai_vision_interpreter"}
        )


# Fallback function for when API key is not available
def create_mock_vision_response(prompt_bundle: PromptBundle) -> GeometrySpec:
    """Create a mock vision response when AI API is not available."""
    
    # Create a simple mock object based on the prompt
    mock_objects = [
        create_geometry_object("circle", center_x=100, center_y=100, radius=50),
        create_geometry_object("line", start_x=50, start_y=50, end_x=150, end_y=150, length=141.42)
    ]
    
    annotations = {
        'ai_vision_confidence': 0.6,
        'detection_method': 'mock_vision',
        'note': 'Mock response - OpenAI API key not configured',
        'prompt_reference': prompt_bundle.prompt_id
    }
    
    return GeometrySpec(
        objects=mock_objects,
        constraints=[],
        annotations=annotations,
        status=Status.VISION,
        confidence=0.6
    ) 