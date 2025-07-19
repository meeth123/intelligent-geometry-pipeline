"""
Symbolic Geometry Planner Agent for Geometry Pipeline.

Enhanced with Gemini 2.5 Pro for intelligent constraint solving and coordinate calculation.
Receives: GeometrySpec with objects and constraints
Produces: CoordinateSolution with exact positions and measurements
"""
import logging
import json
import os
import math
import google.generativeai as genai
from typing import Union, Dict, Any, List, Tuple
from datetime import datetime
from .data_structures import (
    GeometrySpec, CoordinateSolution, AgentError, Status,
    GeometryObject, GeometryConstraint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiSymbolicGeometryPlanner:
    """Uses Gemini 2.5 Pro to solve geometric constraints and calculate precise coordinates."""
    
    def __init__(self):
        self.model_name = "gemini-2.5-pro"  # Gemini 2.5 Pro for mathematical constraint solving
        self.max_tokens = 8000
        
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
            logger.info("Gemini symbolic geometry planner configured successfully")
        else:
            logger.warning("Google API key not available - using fallback constraint solving")
        
        # Create debug directory for JSON logs
        self.debug_dir = "debug_json_responses"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def _save_debug_json(self, response_text: str, prompt_id: str, status: str):
        """Save raw JSON response for debugging purposes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"symbolic_planner_{prompt_id[:8]}_{timestamp}_{status}.json"
            filepath = os.path.join(self.debug_dir, filename)
            
            debug_data = {
                "agent": "symbolic_geometry_planner",
                "prompt_id": prompt_id,
                "timestamp": timestamp,
                "status": status,
                "raw_response": response_text,
                "response_length": len(response_text)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved debug JSON to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug JSON: {e}")
    
    def create_constraint_solving_prompt(self, geometry_spec: GeometrySpec) -> str:
        """Create a structured prompt for Gemini mathematical constraint solving."""
        
        # Extract objects and constraints info
        objects_info = []
        for i, obj in enumerate(geometry_spec.objects):
            obj_info = {
                "index": i,
                "id": obj.id,
                "type": obj.type,
                "properties": dict(obj.properties)
            }
            objects_info.append(obj_info)
        
        constraints_info = []
        for constraint in geometry_spec.constraints:
            constraint_info = {
                "type": constraint.type,
                "objects": constraint.objects,
                "parameters": dict(constraint.parameters) if constraint.parameters else {}
            }
            constraints_info.append(constraint_info)
        
        system_prompt = f"""
You are an expert mathematical constraint solver AND intelligent geometric designer specializing in computational geometry.

Your DUAL ROLE:
1. Make INTELLIGENT ASSUMPTIONS for incomplete specifications
2. Solve mathematical constraints with precision

STEP 1: INTELLIGENT ASSUMPTION-MAKING
Analyze the geometry specification and identify missing information:

OBJECTS (may have incomplete properties):
{json.dumps(objects_info, indent=2)}

CONSTRAINTS (relationships to satisfy):
{json.dumps(constraints_info, indent=2)}

For ANY missing dimensions or unspecified properties, make GEOMETRICALLY INTELLIGENT assumptions:

ASSUMPTION PRINCIPLES:
- **Circles**: Default radius ~5-10 units for good visibility
- **Triangles**: Choose type for best constraint satisfaction:
  * For inscribed circle: Use triangle with integer coordinates for clarity
  * For angle bisectors: Isosceles or equilateral for symmetry
  * Default side length ~10 units
- **Squares/Rectangles**: Default side ~8-10 units for proportion
- **Lines**: Default length ~12 units or as needed for constraints
- **Positioning**: Center constructions at origin for symmetry
- **Optimization**: Choose proportions that create clear, non-overlapping visualization

GEOMETRIC INTELLIGENCE:
- Inscribed circle in triangle: Choose triangle dimensions that produce nice radius values
- Angle bisectors: Use symmetric triangles where bisectors have elegant coordinates  
- Multiple shapes: Scale appropriately so all shapes are clearly visible
- Complex constructions: Optimize for educational clarity and mathematical elegance

STEP 2: MATHEMATICAL CONSTRAINT SOLVING
After making intelligent assumptions, solve constraints precisely:

1. Set up coordinate system (0,0 at center, +x right, +y up)
2. Apply geometric formulas for each constraint
3. Solve system of equations for exact coordinates
4. Calculate all derived properties

CONSTRAINT FORMULAS:
- INSCRIBED: inner shape vertices tangent to outer shape boundary
- TANGENT: perpendicular distance = radius
- ANGLE_BISECTOR: line from vertex through incenter
- PARALLEL/PERPENDICULAR: slope relationships
- EQUAL: matching dimensions/properties
   - DISTANCE: specific spacing between objects

5. COORDINATE CALCULATION:
   - Start with the most constrained object (usually circles or fixed points)
   - Work outward solving for other object positions
   - Ensure all constraints are satisfied mathematically
   - Use exact values when possible (like π, √2, √3)

6. VALIDATION:
   - Verify each constraint is satisfied by the calculated coordinates
   - Check for mathematical consistency
   - Ensure realistic geometry (no overlapping unless intended)

Return a JSON with intelligent assumptions AND complete coordinate solution:
{{
    "reasoning": "Step-by-step process: assumptions made + mathematical derivation",
    "assumptions_made": [
        {{
            "object_id": "object_id",
            "missing_property": "radius|side_length|dimensions|type",
            "assumed_value": "value_chosen",
            "rationale": "why this value was chosen for optimal visualization/constraints"
        }}
    ],
    "coordinate_system": {{
        "origin": "center", 
        "units": "cm",
        "x_axis": "right_positive",
        "y_axis": "up_positive"
    }},
    "solved_objects": [
        {{
            "object_id": "object_id",
            "type": "circle|square|triangle|rectangle|line|point",
            "coordinates": {{
                "center_x": number,
                "center_y": number,
                "radius": number (for circles),
                "vertices": [[x1,y1], [x2,y2], ...] (for polygons),
                "start_point": [x, y] (for lines),
                "end_point": [x, y] (for lines),
                "width": number,
                "height": number,
                "rotation_angle": number (in degrees)
            }},
            "calculated_properties": {{
                "area": number,
                "perimeter": number,
                "bounding_box": {{"min_x": number, "min_y": number, "max_x": number, "max_y": number}}
            }}
        }}
    ],
    "constraint_verification": [
        {{
            "constraint_type": "inscribed|tangent|parallel|perpendicular|equal|distance",
            "objects_involved": ["id1", "id2"],
            "mathematical_check": "equation showing constraint is satisfied",
            "satisfied": true|false,
            "tolerance": number
        }}
    ],
    "solution_quality": {{
        "all_constraints_satisfied": true|false,
        "mathematical_accuracy": "exact|approximate",
        "confidence": 0.0-1.0,
        "alternative_solutions": number
    }}
}}

CRITICAL MATHEMATICAL GUIDELINES:
- Use exact mathematical relationships when possible
- For inscribed polygons, use circumradius formulas
- For tangent relationships, use distance-to-line formulas
- Account for symmetry and optimization (minimize overlaps)
- Provide coordinates with sufficient precision (at least 3 decimal places)
- Always verify your solution satisfies ALL constraints

Focus on mathematical rigor and geometric precision.

IMPORTANT: Your entire response must be ONLY the JSON object. Do not include any text, notes, or markdown formatting like ```json before or after the JSON structure.
"""
        
        return system_prompt
    
    def call_gemini_for_constraint_solving(self, geometry_spec: GeometrySpec) -> Union[Dict[str, Any], AgentError]:
        """Call Gemini to solve geometric constraints mathematically with retry mechanism."""
        
        if not self.api_key:
            return AgentError(
                error="NO_API_KEY",
                message="Google API key not available for constraint solving.",
                details={"agent": "symbolic_geometry_planner"}
            )
        
        max_retries = 3
        geom_id = getattr(geometry_spec, 'spec_id', 'constraint_solving')
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Solving constraints (Attempt {attempt + 1}/{max_retries}) for {len(geometry_spec.objects)} objects with {len(geometry_spec.constraints)} constraints")
                
                prompt = self.create_constraint_solving_prompt(geometry_spec)
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=0.1,  # Low temperature for mathematical precision
                    )
                )
                
                response_text = response.text
                logger.info(f"Gemini constraint solving response: {len(response_text)} characters")
                
                # Save raw response for debugging
                self._save_debug_json(response_text, geom_id, "raw_response")
                
                # Try to extract JSON from the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    
                    # Save successfully parsed JSON
                    self._save_debug_json(json_str, geom_id, "parsed_success")
                    
                    # Add debugging info and return success
                    logger.info(f"Gemini reasoning extracted: {parsed_data.get('reasoning', 'No reasoning')[:100]}...")
                    return parsed_data
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt + 1} JSON parsing failed: {e}")
                
                # Save failed JSON for debugging
                self._save_debug_json(response_text, geom_id, "parse_failed")
                
                # Try to clean the JSON and parse again
                try:
                    cleaned_response = self._clean_json_response(response_text)
                    parsed_data = json.loads(cleaned_response)
                    logger.info("Successfully parsed cleaned JSON response")
                    
                    # Save cleaned successful JSON
                    self._save_debug_json(cleaned_response, geom_id, "cleaned_success")
                    
                    # Add debugging info and return success
                    logger.info(f"Gemini reasoning extracted: {parsed_data.get('reasoning', 'No reasoning')[:100]}...")
                    return parsed_data
                    
                except Exception as cleanup_error:
                    logger.warning(f"JSON cleanup also failed: {cleanup_error}")
                    logger.error(f"Raw response: {response_text[:500]}...")
                    
                    # Save final failed attempt
                    self._save_debug_json(cleaned_response if 'cleaned_response' in locals() else response_text, 
                                        geom_id, "cleanup_failed")
                
                # If this was the last attempt, return error
                if attempt == max_retries - 1:
                    logger.error(f"Symbolic Geometry Planner failed after {max_retries} attempts")
                    return AgentError(
                        error="CONSTRAINT_SOLVING_FAILED",
                        message="The AI failed to solve geometric constraints after multiple attempts. The geometry specification was valid, but coordinate solution could not be computed.",
                        details={"agent": "symbolic_geometry_planner", "final_error": str(e)}
                    )
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with general error: {e}")
                
                # If this was the last attempt, return error
                if attempt == max_retries - 1:
                    logger.error(f"Symbolic Geometry Planner failed after {max_retries} attempts")
                    return AgentError(
                        error="CONSTRAINT_SOLVING_FAILED",
                        message="The AI failed to solve geometric constraints after multiple attempts. The geometry specification was valid, but coordinate solution could not be computed.",
                        details={"agent": "symbolic_geometry_planner", "final_error": str(e)}
                    )
        
        # This should not be reached, but as a safeguard
        return AgentError(
            error="CONSTRAINT_SOLVING_FAILED",
            message="Unknown error in retry loop.",
            details={"agent": "symbolic_geometry_planner"}
        )
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response by removing control characters and fixing common issues."""
        import re
        import json
        
        # First, handle markdown code blocks (```json ... ```)
        code_block_match = re.search(r'```[a-zA-Z]*\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if code_block_match:
            response_text = code_block_match.group(1)
        
        # Remove control characters (except valid JSON whitespace)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', response_text)
        
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix corrupted HTML/SVG closing tags (Issue #3)
        def fix_corrupted_closing_tags(text):
            corrupted_tag_pattern = r'\\([a-zA-Z][a-zA-Z0-9]*)'
            
            def fix_tag(match):
                tag_name = match.group(1)
                remaining_text = text[match.end():]
                if remaining_text.startswith('>'):
                    return f'</{tag_name}'
                else:
                    return match.group(0)
            
            return re.sub(corrupted_tag_pattern, fix_tag, text)
        
        cleaned = fix_corrupted_closing_tags(cleaned)
        
        # If it's still not valid JSON, try a more aggressive approach
        try:
            json.loads(cleaned)
            return cleaned.strip()
        except:
            # More aggressive cleaning for problematic cases
            
            # Extract just the JSON object structure
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                extracted = json_match.group()
                
                # Clean up the extracted JSON more aggressively
                extracted = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', extracted)
                extracted = re.sub(r',(\s*[}\]])', r'\1', extracted)
                
                # Try to fix common issues with multiline strings
                # Replace literal newlines in string values with \n
                in_string = False
                escape_next = False
                result = ""
                
                for char in extracted:
                    if escape_next:
                        result += char
                        escape_next = False
                    elif char == '\\':
                        result += char
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                        result += char
                    elif char == '\n' and in_string:
                        result += '\\n'  # Escape newlines inside strings
                    elif char == '\t' and in_string:
                        result += '\\t'  # Escape tabs inside strings  
                    elif char == '\r' and in_string:
                        result += '\\r'  # Escape carriage returns inside strings
                    else:
                        result += char
                
                return result.strip()
        
        return cleaned.strip()
    
    def create_fallback_solution(self, geometry_spec: GeometrySpec) -> Dict[str, Any]:
        """Create a basic fallback solution when Gemini is not available."""
        
        solved_objects = []
        x_offset = 0
        
        for obj in geometry_spec.objects:
            if obj.type == "circle":
                radius = obj.properties.get('radius', 50)
                solved_obj = {
                    "object_id": obj.id,
                    "type": "circle",
                    "coordinates": {
                        "center_x": x_offset,
                        "center_y": 0,
                        "radius": radius
                    },
                    "calculated_properties": {
                        "area": math.pi * radius**2,
                        "perimeter": 2 * math.pi * radius
                    }
                }
            elif obj.type == "square":
                side = obj.properties.get('side_length', 100)
                half_side = side / 2
                solved_obj = {
                    "object_id": obj.id,
                    "type": "square",
                    "coordinates": {
                        "center_x": x_offset,
                        "center_y": 0,
                        "vertices": [
                            [x_offset - half_side, -half_side],
                            [x_offset + half_side, -half_side],
                            [x_offset + half_side, half_side],
                            [x_offset - half_side, half_side]
                        ],
                        "width": side,
                        "height": side
                    },
                    "calculated_properties": {
                        "area": side**2,
                        "perimeter": 4 * side
                    }
                }
            else:
                # Generic fallback for other shapes
                solved_obj = {
                    "object_id": obj.id,
                    "type": obj.type,
                    "coordinates": {"center_x": x_offset, "center_y": 0},
                    "calculated_properties": {"area": 0, "perimeter": 0}
                }
            
            solved_objects.append(solved_obj)
            x_offset += 150  # Space objects apart
        
        return {
            "reasoning": f"Fallback geometric solution. Positioned {len(solved_objects)} objects with basic spacing.",
            "coordinate_system": {"origin": "center", "units": "pixels"},
            "solved_objects": solved_objects,
            "constraint_verification": [],
            "solution_quality": {
                "all_constraints_satisfied": False,
                "mathematical_accuracy": "approximate",
                "confidence": 0.5
            }
        }
    
    def convert_to_coordinate_solution(self, gemini_response: Dict[str, Any], 
                                     original_spec: GeometrySpec) -> CoordinateSolution:
        """Convert Gemini's response to a CoordinateSolution object."""
        
        # Extract coordinate data
        solved_objects = gemini_response.get("solved_objects", [])
        coordinate_system = gemini_response.get("coordinate_system", {})
        
        # Create coordinate solution
        solution = CoordinateSolution(
            coordinate_system=coordinate_system,
            object_coordinates={},
            constraint_solutions={},
            mathematical_derivation=gemini_response.get("reasoning", "No reasoning provided"),
            accuracy_metrics={
                "precision": "high" if gemini_response.get("solution_quality", {}).get("mathematical_accuracy") == "exact" else "medium",
                "all_constraints_satisfied": gemini_response.get("solution_quality", {}).get("all_constraints_satisfied", False),
                "confidence": gemini_response.get("solution_quality", {}).get("confidence", 0.5)
            },
            status=Status.SOLVED
        )
        
        # Populate object coordinates
        for solved_obj in solved_objects:
            obj_id = solved_obj.get("object_id")
            coordinates = solved_obj.get("coordinates", {})
            properties = solved_obj.get("calculated_properties", {})
            
            solution.object_coordinates[obj_id] = {
                **coordinates,
                **properties
            }
        
        # Populate constraint solutions
        constraint_verifications = gemini_response.get("constraint_verification", [])
        for i, verification in enumerate(constraint_verifications):
            solution.constraint_solutions[f"constraint_{i}"] = {
                "type": verification.get("constraint_type"),
                "satisfied": verification.get("satisfied", False),
                "mathematical_proof": verification.get("mathematical_check", ""),
                "tolerance": verification.get("tolerance", 0.001)
            }
        
        return solution


# Global planner instance
_planner = GeminiSymbolicGeometryPlanner()


def handle(geometry_spec: GeometrySpec) -> Union[CoordinateSolution, AgentError]:
    """
    Main entry point for the symbolic geometry planner agent.
    
    Args:
        geometry_spec: GeometrySpec with objects and constraints to solve
        
    Returns:
        CoordinateSolution with precise coordinates and measurements or AgentError
    """
    try:
        logger.info("Symbolic Geometry Planner agent starting...")
        
        if not isinstance(geometry_spec, GeometrySpec):
            return AgentError(
                error="INVALID_INPUT",
                message="Input must be a GeometrySpec"
            )
        
        if not geometry_spec.objects:
            return AgentError(
                error="NO_OBJECTS",
                message="GeometrySpec must contain at least one object"
            )
        
        logger.info(f"Solving constraints for {len(geometry_spec.objects)} objects with {len(geometry_spec.constraints)} constraints")
        
        # Solve constraints with Gemini
        gemini_response = _planner.call_gemini_for_constraint_solving(geometry_spec)
        
        # Check if constraint solving failed
        if isinstance(gemini_response, AgentError):
            return gemini_response
        
        # Convert to CoordinateSolution
        coordinate_solution = _planner.convert_to_coordinate_solution(gemini_response, geometry_spec)
        
        # Add Gemini reasoning
        coordinate_solution.agent_reasoning = {"symbolic_geometry_planner": gemini_response.get("reasoning", "No reasoning provided")}
        
        logger.info("Symbolic Geometry Planner completed successfully")
        return coordinate_solution
        
    except Exception as e:
        logger.error(f"Error in symbolic geometry planner: {e}")
        return AgentError(
            error="PLANNING_ERROR",
            message=str(e),
            details={"agent": "symbolic_geometry_planner"}
        ) 