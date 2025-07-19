"""
AI-powered Prompt Interpreter Agent for Geometry Pipeline.

Enhanced with Gemini 2.5 Pro for intelligent geometric reasoning.
Receives: PromptBundle
Produces: GeometrySpec with geometric objects and relationships
"""
import logging
import json
import os
import google.generativeai as genai
from typing import Union, Dict, Any, List
from .data_structures import (
    PromptBundle, GeometrySpec, GeometryObject, GeometryConstraint,
    Status, AgentError, create_geometry_object, create_constraint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiPromptInterpreter:
    """Uses Gemini 2.5 flash to interpret text prompts and extract geometric specifications."""
    
    def __init__(self):
        self.model_name = "gemini-2.5-flash"  # Gemini 2.5 flash with superior reasoning capabilities
        self.max_tokens = 8000  # Gemini optimal token count
        
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
            logger.info("Gemini model configured successfully")
        else:
            logger.warning("Google API key not available - using fallback analysis")
    
    def create_geometry_analysis_prompt(self, user_prompt: str) -> str:
        """Create a structured prompt for Gemini geometric reasoning."""
        
        system_prompt = """
You are an expert geometric analyst with deep mathematical reasoning capabilities. 

THINK STEP BY STEP through this geometric problem:

1. PARSE the user's prompt carefully
   - What geometric objects are mentioned?
   - What are the explicit dimensions and measurements?
   - What spatial relationships are described?

2. REASON about geometric constraints
   - "fits inside" + "vertices lie on" = INSCRIBED polygon with vertices tangent to circle
   - "circle of 10cm" typically means diameter=10cm, so radius=5cm
   - "square inscribed in circle" means all 4 vertices touch the circle circumference
   - Look for: parallel, perpendicular, equal, tangent, inscribed, circumscribed relationships

3. CALCULATE correct dimensions
   - For inscribed square in circle: diagonal = diameter, side = diameter/√2
   - Convert units consistently (cm, mm, pixels, etc.)
   - Preserve the user's specified measurements

4. STRUCTURE the output as JSON

Return a JSON object with this structure:
{
    "reasoning": "Step-by-step analysis of the geometric problem",
    "objects": [
        {
            "type": "circle|triangle|square|rectangle|line|polygon|point",
            "properties": {
                "radius": number (for circles),
                "side_length": number (for squares/triangles),
                "width": number (for rectangles),
                "height": number (for rectangles), 
                "length": number (for lines),
                "center_x": number (optional, default 0),
                "center_y": number (optional, default 0)
            }
        }
    ],
    "constraints": [
        {
            "type": "inscribed|tangent|parallel|perpendicular|equal|centered|distance|intersects",
            "object_indices": [0, 1],
            "description": "detailed explanation of the constraint",
            "parameters": {
                "relationship": "specific geometric relationship"
            }
        }
    ],
    "extracted_info": {
        "dimensions": {"original_measurements": "as_specified_by_user"},
        "angles": {...},
        "style_hints": [...],
        "geometric_intent": "high-level description of what user wants"
    }
}

CRITICAL: Pay special attention to geometric relationships like:
- "vertices lie on circle" = tangent constraint between polygon vertices and circle
- "fits inside" = inscribed constraint
- "surrounds" = circumscribed constraint
- "parallel lines" = parallel constraint
- "right angles" = perpendicular constraint

Be mathematically precise with dimensions and relationships.
"""
        
        return f"{system_prompt}\n\nUser prompt to analyze: \"{user_prompt}\""
    
    def call_gemini_for_analysis(self, user_prompt: str) -> Dict[str, Any]:
        """Call Gemini to analyze the geometric prompt."""
        
        if not self.api_key:
            # Return a basic fallback analysis
            return self.create_fallback_analysis(user_prompt)
        
        try:
            prompt = self.create_geometry_analysis_prompt(user_prompt)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.1,
                )
            )
            
            response_text = response.text
            logger.info(f"Gemini Response received: {len(response_text)} characters")
            
            # Store the raw Gemini reasoning for frontend display
            raw_reasoning = response_text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Gemini JSON response: {e}")
                logger.error(f"Raw response: {response_text[:500]}...")
                
                # Create fallback spec but preserve the raw reasoning
                parsed_data = {
                    "reasoning": f"Gemini provided reasoning but JSON parsing failed. Raw response: {response_text[:300]}...",
                    "objects": self.create_fallback_analysis(user_prompt)["objects"],
                    "constraints": self.create_fallback_analysis(user_prompt)["constraints"],
                    "extracted_info": {"error": "Failed to parse Gemini JSON but got response"}
                }
            
            # Extract Gemini reasoning for display
            gemini_reasoning = parsed_data.get("reasoning", "No reasoning provided")
            
            # Debug logging
            logger.info(f"Gemini reasoning extracted: {gemini_reasoning[:100]}...")
            logger.info(f"Parsed data keys: {list(parsed_data.keys())}")
            
            # Add raw response for debugging
            parsed_data["raw_gemini_reasoning"] = raw_reasoning[:1000]  # Truncate for storage
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self.create_fallback_analysis(user_prompt)
    
    def create_fallback_analysis(self, user_prompt: str) -> Dict[str, Any]:
        """Create a basic fallback analysis when AI is not available."""
        
        # Simple keyword-based fallback
        prompt_lower = user_prompt.lower()
        
        objects = []
        constraints = []
        
        # Basic shape detection
        if "circle" in prompt_lower:
            objects.append({"type": "circle", "properties": {"radius": 50}})
        if "square" in prompt_lower:
            objects.append({"type": "square", "properties": {"side_length": 100}})
        if "triangle" in prompt_lower:
            objects.append({"type": "triangle", "properties": {"side_length": 100}})
        if "line" in prompt_lower:
            objects.append({"type": "line", "properties": {"length": 100}})
        
        # Basic constraint detection
        if any(word in prompt_lower for word in ["inside", "inscribed", "fits"]):
            if len(objects) >= 2:
                constraints.append({
                    "type": "inscribed",
                    "object_indices": [0, 1],
                    "description": "One shape inside another",
                    "parameters": {"relationship": "inside"}
                })
        
        return {
            "reasoning": f"Fallback analysis used due to missing API key. Detected {len(objects)} objects and {len(constraints)} constraints using rule-based pattern matching.",
            "objects": objects,
            "constraints": constraints,
            "extracted_info": {
                "dimensions": {},
                "angles": {},
                "style_hints": []
            }
        }
    
    def parse_ai_analysis(self, ai_response: Dict[str, Any]) -> tuple[List[GeometryObject], List[GeometryConstraint]]:
        """Parse AI analysis into GeometryObjects and GeometryConstraints."""
        
        objects = []
        constraints = []
        
        # Create geometry objects
        for obj_data in ai_response.get("objects", []):
            obj_type = obj_data.get("type", "unknown")
            properties = obj_data.get("properties", {})
            
            geometry_obj = create_geometry_object(obj_type, **properties)
            objects.append(geometry_obj)
        
        # Create constraints
        for constraint_data in ai_response.get("constraints", []):
            constraint_type = constraint_data.get("type", "unknown")
            object_indices = constraint_data.get("object_indices", [])
            parameters = constraint_data.get("parameters", {})
            description = constraint_data.get("description", "")
            
            # Map indices to object IDs
            if len(object_indices) >= 2 and all(isinstance(i, int) and i < len(objects) for i in object_indices):
                try:
                    object_ids = [objects[i].id for i in object_indices]
                    if len(object_ids) >= 2:
                        constraint = create_constraint(
                            constraint_type, 
                            object_ids, 
                            description=description,
                            **parameters
                        )
                        constraints.append(constraint)
                except (IndexError, TypeError):
                    logger.warning(f"Invalid object indices in constraint: {object_indices}")
        
        return objects, constraints
    
    def process_prompt(self, prompt_bundle: PromptBundle) -> GeometrySpec:
        """Process a prompt bundle using AI analysis."""
        
        logger.info(f"AI analyzing prompt: {prompt_bundle.text[:100]}...")
        
        # Get AI analysis (this now includes reasoning)
        ai_response = self.call_gemini_for_analysis(prompt_bundle.text)
        
        # Parse into geometry objects and constraints
        objects, constraints = self.parse_ai_analysis(ai_response)
        
        # Create annotations with AI analysis info
        extracted_info = ai_response.get("extracted_info", {})
        annotations = {
            'ai_analysis': True,
            'extracted_dimensions': extracted_info.get('dimensions', {}),
            'extracted_angles': extracted_info.get('angles', {}),
            'style_hints': extracted_info.get('style_hints', []),
            'original_text': prompt_bundle.text,
            'ai_model': self.model_name
        }
        
        # Create GeometrySpec with O3 reasoning
        spec = GeometrySpec(
            objects=objects,
            constraints=constraints,
            annotations=annotations,
            status=Status.DRAFT
        )
        
        # Add O3 reasoning to the spec
        gemini_reasoning = ai_response.get("reasoning", "No reasoning provided")
        spec.agent_reasoning["prompt_interpreter"] = gemini_reasoning
        spec.processing_steps.append({
            "agent": "prompt_interpreter", 
            "step": "ai_analysis",
            "reasoning": gemini_reasoning,
            "model": self.model_name,
            "objects_found": len(objects),
            "constraints_found": len(constraints)
        })
        
        logger.info(f"AI generated spec with {len(objects)} objects, {len(constraints)} constraints")
        return spec


# Global AI interpreter instance
_ai_interpreter = GeminiPromptInterpreter()


def handle(prompt_bundle: PromptBundle) -> Union[GeometrySpec, AgentError]:
    """
    Main entry point for the AI prompt interpreter agent.
    
    Args:
        prompt_bundle: PromptBundle containing user input
        
    Returns:
        GeometrySpec with AI-extracted geometric information or AgentError
    """
    try:
        logger.info("AI Prompt interpreter agent starting...")
        
        if not isinstance(prompt_bundle, PromptBundle):
            return AgentError(
                error="INVALID_INPUT",
                message="Input must be a PromptBundle"
            )
        
        if not prompt_bundle.text.strip():
            return AgentError(
                error="UNRECOGNIZED_TEXT",
                message="Prompt text cannot be empty"
            )
        
        # Process with AI
        geometry_spec = _ai_interpreter.process_prompt(prompt_bundle)
        
        # Ensure we have at least one object
        if not geometry_spec.objects:
            # Create a default line object
            default_obj = create_geometry_object("line", length=100.0)
            geometry_spec.objects.append(default_obj)
        
        logger.info("AI Prompt interpreter completed successfully")
        return geometry_spec
        
    except Exception as e:
        logger.error(f"Error in AI prompt interpreter: {e}")
        return AgentError(
            error="PROCESSING_ERROR",
            message=str(e),
            details={"agent": "ai_prompt_interpreter"}
        ) 