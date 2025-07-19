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
You are an expert mathematical constraint solver specializing in computational geometry.

THINK STEP BY STEP through this constraint solving problem:

1. ANALYZE the geometric objects and their properties:
{json.dumps(objects_info, indent=2)}

2. ANALYZE the constraints between objects:
{json.dumps(constraints_info, indent=2)}

3. MATHEMATICAL REASONING:
   - Set up a coordinate system (0,0 at center, positive x right, positive y up)
   - For each constraint, write the mathematical equations
   - Solve the system of equations to find exact coordinates
   - Calculate all derived properties (areas, perimeters, angles)

4. CONSTRAINT-SPECIFIC FORMULAS:
   - INSCRIBED square in circle: diagonal = diameter, side = diameter/√2
   - TANGENT: distance from center to tangent line = radius
   - PARALLEL: same slope or perpendicular slopes differ by π/2
   - PERPENDICULAR: dot product of direction vectors = 0
   - EQUAL: same dimensions/properties
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

Return a JSON with the complete coordinate solution:
{{
    "reasoning": "Step-by-step mathematical derivation of the solution",
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
"""
        
        return system_prompt
    
    def call_gemini_for_constraint_solving(self, geometry_spec: GeometrySpec) -> Dict[str, Any]:
        """Call Gemini to solve geometric constraints mathematically."""
        
        if not self.api_key:
            return self.create_fallback_solution(geometry_spec)
        
        try:
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
            geom_id = getattr(geometry_spec, 'spec_id', 'constraint_solving')
            self._save_debug_json(response_text, geom_id, "raw_response")
            
            # Try to extract JSON from the response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    
                    # Save successfully parsed JSON
                    self._save_debug_json(json_str, geom_id, "parsed_success")
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Gemini constraint solving JSON: {e}")
                
                # Save failed JSON for debugging
                self._save_debug_json(response_text, geom_id, "parse_failed")
                
                # Try to clean the JSON and parse again
                try:
                    cleaned_response = self._clean_json_response(response_text)
                    parsed_data = json.loads(cleaned_response)
                    logger.info("Successfully parsed cleaned JSON response")
                    
                    # Save cleaned successful JSON
                    self._save_debug_json(cleaned_response, geom_id, "cleaned_success")
                    
                except Exception as cleanup_error:
                    logger.warning(f"JSON cleanup also failed: {cleanup_error}")
                    logger.error(f"Raw response: {response_text[:500]}...")
                    
                    # Save final failed attempt
                    self._save_debug_json(cleaned_response if 'cleaned_response' in locals() else response_text, 
                                        geom_id, "cleanup_failed")
                    
                    # Create fallback with raw reasoning
                    parsed_data = {
                        "reasoning": f"Gemini provided mathematical reasoning but JSON parsing failed. Raw response: {response_text[:300]}...",
                        "coordinate_system": {"origin": "center", "units": "cm"},
                        "solved_objects": self.create_fallback_solution(geometry_spec)["solved_objects"],
                        "constraint_verification": [],
                        "solution_quality": {"all_constraints_satisfied": False, "confidence": 0.3}
                    }
            
            # Add debugging info
            logger.info(f"Gemini reasoning extracted: {parsed_data.get('reasoning', 'No reasoning')[:100]}...")
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Gemini constraint solving failed: {e}")
            return self.create_fallback_solution(geometry_spec)
    
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