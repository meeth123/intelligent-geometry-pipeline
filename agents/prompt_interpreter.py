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
    Status, AgentError, ClarificationRequest, create_geometry_object, create_constraint
)
from datetime import datetime

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
        
        # Create debug directory for JSON logs
        self.debug_dir = "debug_json_responses"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def _save_debug_json(self, response_text: str, prompt_id: str, status: str):
        """Save raw JSON response for debugging purposes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_interpreter_{prompt_id[:8]}_{timestamp}_{status}.json"
            filepath = os.path.join(self.debug_dir, filename)
            
            debug_data = {
                "agent": "prompt_interpreter",
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
    
    def create_geometry_analysis_prompt(self, user_prompt: str) -> str:
        """Create a structured prompt for Gemini intent extraction (NO assumptions)."""
        
        system_prompt = """
You are an expert geometry teacher focused on understanding user INTENT without making assumptions.

Your job: Extract what the user wants WITHOUT making specific geometric assumptions or calculations.

ANALYZE the user's prompt to identify:

1. GEOMETRIC OBJECTS mentioned (what shapes/objects are requested)
2. EXPLICIT RELATIONSHIPS described (how objects relate to each other) 
3. EXPLICIT MEASUREMENTS provided by user (preserve exactly as stated)
4. AMBIGUITIES and missing information (what needs to be determined later)

DO NOT:
- Make up specific dimensions if not provided
- Calculate exact coordinates or measurements  
- Assume specific triangle types (equilateral, isosceles, etc.)
- Choose specific positioning or orientations
- Make mathematical calculations or conversions

DO:
- Identify the types of shapes mentioned
- Extract explicitly stated relationships
- Preserve user's exact measurements when given
- Note what information is missing or ambiguous

Return a JSON object with this structure:
{
    "reasoning": "What the user wants and what information they provided vs. what's missing",
    "objects": [
        {
            "type": "circle|triangle|square|rectangle|line|polygon|point", 
            "properties": {
                "user_specified_radius": number (ONLY if user provided),
                "user_specified_side": number (ONLY if user provided),
                "user_specified_width": number (ONLY if user provided),
                "user_specified_height": number (ONLY if user provided),
                "name": "descriptive name from user"
            }
        }
    ],
    "constraints": [
        {
            "type": "inscribed|tangent|parallel|perpendicular|equal|intersects|angle_bisector",
            "object_indices": [0, 1],
            "description": "relationship described by user",
            "parameters": {
                "relationship": "user's description",
                "incomplete": true/false
            }
        }
    ],
    "extracted_info": {
        "explicit_measurements": {"measurements user actually provided"},
        "missing_information": ["what needs to be determined by symbolic planner"],
        "geometric_intent": "high-level description of what user wants to see",
        "ambiguities": ["unclear aspects that need geometric reasoning"]
    }
}

Focus on INTENT EXTRACTION, not assumption-making. Let the symbolic planner handle the mathematical reasoning.
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
            
            # Save raw response for debugging
            self._save_debug_json(response_text, "gemini_analysis", "raw_response")
            
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
                    
                    # Save successfully parsed JSON
                    self._save_debug_json(json_str, "gemini_analysis", "parsed_success")
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Gemini JSON response: {e}")
                
                # Save failed JSON for debugging
                self._save_debug_json(response_text, "gemini_analysis", "parse_failed")
                
                # Try to clean the JSON and parse again
                try:
                    cleaned_response = _clean_json_response(response_text)
                    parsed_data = json.loads(cleaned_response)
                    logger.info("Successfully parsed cleaned Gemini JSON response")
                    
                    # Save cleaned successful JSON
                    self._save_debug_json(cleaned_response, "gemini_analysis", "cleaned_success")
                    
                except Exception as cleanup_error:
                    logger.warning(f"Gemini JSON cleanup also failed: {cleanup_error}")
                    
                    # Save final failed attempt
                    self._save_debug_json(cleaned_response if 'cleaned_response' in locals() else response_text, 
                                        "gemini_analysis", "cleanup_failed")
                    
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
        """Create a basic fallback analysis when AI is not available - NO assumptions."""
        
        # Simple keyword-based fallback - only extract what's mentioned
        prompt_lower = user_prompt.lower()
        
        objects = []
        constraints = []
        missing_info = []
        
        # Basic shape detection - NO default dimensions
        if "circle" in prompt_lower:
            objects.append({"type": "circle", "properties": {"name": "circle"}})
            missing_info.append("circle dimensions not specified")
        if "square" in prompt_lower:
            objects.append({"type": "square", "properties": {"name": "square"}})
            missing_info.append("square dimensions not specified")
        if "triangle" in prompt_lower:
            objects.append({"type": "triangle", "properties": {"name": "triangle"}})
            missing_info.append("triangle type and dimensions not specified")
        if "line" in prompt_lower:
            objects.append({"type": "line", "properties": {"name": "line"}})
            missing_info.append("line length not specified")
        
        # Basic constraint detection - identify relationships only
        if any(word in prompt_lower for word in ["inside", "inscribed", "fits"]):
            if len(objects) >= 2:
                constraints.append({
                    "type": "inscribed",
                    "object_indices": [0, 1],
                    "description": "One shape inside another (relationship needs mathematical definition)",
                    "parameters": {"relationship": "inscribed", "incomplete": True}
                })
        
        if any(word in prompt_lower for word in ["bisector", "bisect"]):
            missing_info.append("angle bisector relationships need geometric calculation")
        
        return {
            "reasoning": f"Fallback analysis: Identified {len(objects)} objects and {len(constraints)} relationships. All specific dimensions and positioning need to be determined by symbolic planner.",
            "objects": objects,
            "constraints": constraints,
            "extracted_info": {
                "explicit_measurements": {},
                "missing_information": missing_info,
                "geometric_intent": "Create geometric construction based on identified objects and relationships",
                "ambiguities": ["All dimensions and positioning unspecified"]
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
            # Support both single-object constraints (orientation, scaling) and multi-object constraints (relative positioning)
            if len(object_indices) >= 1 and all(isinstance(i, int) and i < len(objects) for i in object_indices):
                try:
                    object_ids = [objects[i].id for i in object_indices]
                    constraint = create_constraint(
                        constraint_type, 
                        object_ids, 
                        description=description,
                        **parameters
                    )
                    constraints.append(constraint)
                    logger.info(f"Created constraint: {constraint_type} for objects {object_ids}")
                except (IndexError, TypeError) as e:
                    logger.warning(f"Invalid object indices in constraint: {object_indices}, error: {e}")
            else:
                logger.warning(f"Skipping constraint with invalid indices: {object_indices}")
        
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


def _clean_json_response(response_text: str) -> str:
    """Enhanced JSON response cleaning with specialized fixes for common AI response issues."""
    import re
    import json
    
    # Stage 1: Extract JSON from markdown code blocks
    code_block_match = re.search(r'```[a-zA-Z]*\s*(\{.*\})\s*```', response_text, re.DOTALL)
    if code_block_match:
        response_text = code_block_match.group(1)
    
    # Stage 2: Remove control characters (except valid JSON whitespace)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', response_text)
    
    # Stage 3: Enhanced fixes for specific common issues
    
    # Fix 1: Missing commas between array objects (Issue #1)
    # Pattern: } followed by whitespace and { without a comma
    # Also handle cases with newlines and varied whitespace
    cleaned = re.sub(r'}(\s*\n\s*)\{', r'},\1{', cleaned)  # Handle newlines
    cleaned = re.sub(r'}(\s+)\{', r'},\1{', cleaned)  # Handle spaces/tabs
    
    # More aggressive: find closing braces followed by opening braces in arrays
    # Look for patterns like:  }  \n    }  \n    {  (missing comma after first })
    lines = cleaned.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # If this line is just a closing brace and the next non-empty line starts an object
        if stripped == '}' and i + 1 < len(lines):
            # Look ahead to find next non-empty line
            next_line_idx = i + 1
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            
            if next_line_idx < len(lines):
                next_line = lines[next_line_idx].strip()
                if next_line == '{':
                    # This looks like missing comma - add it
                    fixed_lines.append(line.rstrip() + ',')
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    cleaned = '\n'.join(fixed_lines)
    
    # Fix 2: SVG quote escaping (Issue #2) 
    # Find SVG fields and properly escape quotes in SVG content
    def fix_svg_quotes(match):
        svg_content = match.group(1)  # The SVG content
        
        # First unescape any existing escapes to normalize
        unescaped = svg_content.replace('\\"', '"')
        # Then properly escape all quotes
        properly_escaped = unescaped.replace('"', '\\"')
        
        return f'"{match.string[match.start():match.start(1)]}"{properly_escaped}"{match.string[match.end(1):match.end()]}"'
    
    # Apply SVG quote fixing to common SVG fields
    # Enhanced SVG quote fixing for complex content with inconsistent escaping
    def fix_svg_field_quotes(text, field_name):
        # Find the field and its content more carefully
        field_pattern = f'"{field_name}"\\s*:\\s*"'
        match = re.search(field_pattern, text)
        if not match:
            return text
        
        content_start = match.end() - 1  # Position of opening quote
        
        # Find the end of the field by parsing character by character
        # and tracking escape sequences properly
        i = content_start + 1  # Start after opening quote
        escape_level = 0
        content_end = i
        
        while i < len(text):
            char = text[i]
            
            if char == '\\':
                # Count consecutive backslashes
                backslash_count = 0
                while i < len(text) and text[i] == '\\':
                    backslash_count += 1
                    i += 1
                
                # If followed by a quote, check if it's properly escaped
                if i < len(text) and text[i] == '"':
                    if backslash_count % 2 == 1:
                        # Odd number of backslashes = escaped quote (continue)
                        i += 1
                        continue
                    else:
                        # Even number of backslashes = unescaped quote
                        # Check if this ends the field
                        remaining = text[i+1:].strip()
                        if remaining.startswith(',') or remaining.startswith('}'):
                            content_end = i
                            break
                        # Otherwise it's an unescaped quote within content
                        i += 1
                        continue
                # Continue with other characters after backslashes
                continue
                
            elif char == '"':
                # Direct unescaped quote - check if it ends the field
                remaining = text[i+1:].strip()
                if remaining.startswith(',') or remaining.startswith('}'):
                    content_end = i
                    break
                # Otherwise it's an unescaped quote within content
                i += 1
                continue
            else:
                i += 1
        
        if content_end > content_start + 1:
            # Extract the SVG content (between quotes)
            svg_content = text[content_start + 1:content_end]
            
            # Normalize quotes: remove all existing escaping, then properly escape
            # This handles mixed escaping scenarios
            normalized = svg_content
            
            # Remove double escaping first (\\\\" -> \\")
            normalized = normalized.replace('\\\\"', '\\"')
            # Remove single escaping (\\") -> ")
            normalized = normalized.replace('\\"', '"')
            
            # Now properly escape all quotes for JSON
            fixed_content = normalized.replace('"', '\\"')
            
            # Reconstruct the text
            return text[:content_start + 1] + fixed_content + text[content_end:]
        
        return text
    
    # Apply the fix to common SVG fields
    for svg_field in ['optimized_svg', 'svg_content', 'render_svg']:
        cleaned = fix_svg_field_quotes(cleaned, svg_field)
    
    # Fix 3: Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix 4: Corrupted HTML/SVG closing tags (Issue #3)
    # Pattern: \tagname> should be </tagname>
    # Examples: \g> -> </g>, \svg> -> </svg>, \defs> -> </defs>
    def fix_corrupted_closing_tags(text):
        # Find patterns like \word> and convert to </word>
        corrupted_tag_pattern = r'\\([a-zA-Z][a-zA-Z0-9]*)'
        
        def fix_tag(match):
            tag_name = match.group(1)
            # Only fix if this looks like an HTML/SVG tag and is followed by >
            remaining_text = text[match.end():]
            if remaining_text.startswith('>'):
                return f'</{tag_name}'
            else:
                # Not a corrupted tag, leave as is
                return match.group(0)
        
        return re.sub(corrupted_tag_pattern, fix_tag, text)
    
    cleaned = fix_corrupted_closing_tags(cleaned)
    
    # Stage 4: Mathematical Expression Evaluation (Issue #4) 
    # Convert JavaScript-style mathematical expressions to literal numbers
    def evaluate_math_expressions(text):
        import math
        import re
        
        # Find mathematical expressions in JSON values
        # Pattern: "field": expression, or "field": expression}
        math_patterns = [
            # Handle Math.sqrt() calls first
            (r'Math\.sqrt\((\d+(?:\.\d+)?)\)', lambda m: str(math.sqrt(float(m.group(1))))),
            # Handle complex arithmetic with sqrt: 40 / (5 + Math.sqrt(89))
            (r'(\d+(?:\.\d+)?)\s*/\s*\(\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*\)', 
             lambda m: str(float(m.group(1)) / (float(m.group(2)) + float(m.group(3))))),
            # Handle simple division: num / num  
            (r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', 
             lambda m: str(float(m.group(1)) / float(m.group(2)))),
            # Handle simple addition: num + num
            (r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', 
             lambda m: str(float(m.group(1)) + float(m.group(2)))),
            # Handle simple subtraction: num - num
            (r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', 
             lambda m: str(float(m.group(1)) - float(m.group(2)))),
        ]
        
        # Apply mathematical expression patterns iteratively
        max_iterations = 5  # Prevent infinite loops
        for iteration in range(max_iterations):
            original_text = text
            
            for pattern, replacement in math_patterns:
                text = re.sub(pattern, replacement, text)
            
            # If no changes made, we're done
            if text == original_text:
                break
        
        return text
    
    cleaned = evaluate_math_expressions(cleaned)

    # Stage 5: Fix structural completeness issues
    # Check for incomplete JSON structures and complete them
    open_braces = cleaned.count('{')
    close_braces = cleaned.count('}')
    open_brackets = cleaned.count('[')
    close_brackets = cleaned.count(']')
    
    # Add missing closing braces and brackets
    missing_braces = open_braces - close_braces
    missing_brackets = open_brackets - close_brackets
    
    if missing_braces > 0 or missing_brackets > 0:
        # Add missing closures to complete the JSON
        for _ in range(missing_brackets):
            cleaned += '\n    ]'
        for _ in range(missing_braces):
            cleaned += '\n}'
    
    # Check for unclosed strings (odd number of unescaped quotes)
    quote_count = 0
    i = 0
    while i < len(cleaned):
        if cleaned[i] == '\\':
            i += 2  # Skip escaped character
            continue
        elif cleaned[i] == '"':
            quote_count += 1
        i += 1
    
    # If odd number of quotes, we have an unclosed string - close it
    if quote_count % 2 == 1:
        cleaned += '"'
    
    # Stage 6: Test if fixes worked
    try:
        json.loads(cleaned)
        return cleaned.strip()
    except:
        # Stage 7: More aggressive cleaning for complex cases
        
        # Extract just the JSON object structure
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            extracted = json_match.group()
            
            # Additional aggressive cleaning
            extracted = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', extracted)
            extracted = re.sub(r',(\s*[}\]])', r'\1', extracted)
            
            # Fix multiline string issues character by character
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


def _ai_detect_logical_issues(prompt_text: str) -> Union[ClarificationRequest, None]:
    """
    Use Gemini AI to detect logical inconsistencies, ambiguities, and contradictions in geometry prompts.
    
    Args:
        prompt_text: The user's input prompt
        
    Returns:
        ClarificationRequest if issues found, None otherwise
    """
    try:
        analysis_prompt = f"""
You are an expert geometry teacher analyzing a student's geometry problem description for logical inconsistencies, contradictions, and ambiguities.

STUDENT PROMPT TO ANALYZE:
"{prompt_text}"

Your task: Carefully analyze this geometry prompt and identify any logical problems that would make the construction impossible, ambiguous, or contradictory.

Look for these types of issues:

1. MATHEMATICAL CONTRADICTIONS:
   - Measurements that don't work together (e.g., "square with 10cm sides inscribed in 5cm circle")
   - Impossible angle relationships
   - Conflicting dimensions

2. LOGICAL INCONSISTENCIES:
   - "Inside" vs "outside" contradictions
   - "Touching" vs "not touching" conflicts
   - Parallel vs perpendicular contradictions
   - Same object described with conflicting properties

3. AMBIGUOUS REFERENCES:
   - Unclear pronoun references ("it", "them", "this")
   - Multiple objects with same name but using "the object"
   - Vague positional descriptions

4. MISSING CRITICAL INFORMATION:
   - No dimensions specified
   - Vague spatial relationships ("near", "close to")
   - Missing reference points for directions

5. IMPOSSIBLE CONSTRUCTIONS:
   - Geometrically impossible relationships
   - Invalid measurements (negative, zero)
   - Angle sums that violate geometric rules

6. TERMINOLOGY INCONSISTENCIES:
   - Mixed 2D/3D terms
   - Incorrect geometric vocabulary
   - Contradictory measurement terms

Return your analysis as JSON in this exact format:
{{
    "has_issues": true/false,
    "issue_category": "MATHEMATICAL_CONTRADICTION" | "LOGICAL_INCONSISTENCY" | "AMBIGUOUS_REFERENCE" | "MISSING_INFORMATION" | "IMPOSSIBLE_CONSTRUCTION" | "TERMINOLOGY_INCONSISTENCY" | "CLEAR",
    "detected_issues": [
        "Brief description of issue 1",
        "Brief description of issue 2"
    ],
    "clarification_questions": [
        "Specific question to resolve issue 1",
        "Specific question to resolve issue 2"  
    ],
    "suggested_resolutions": [
        "Concrete suggestion to fix issue 1",
        "Concrete suggestion to fix issue 2"
    ],
    "reasoning": "Your step-by-step analysis of why these are issues and how they affect the geometry construction"
}}

If the prompt is clear and has no logical issues, return {{"has_issues": false, "issue_category": "CLEAR"}}.

Focus on logic and clarity - don't be overly picky about minor wording. Only flag real issues that would prevent accurate geometry construction.
"""

        # Use Gemini to analyze the prompt
        response = _ai_interpreter.model.generate_content(analysis_prompt)
        response_text = response.text.strip()
        
        logger.info(f"AI logical analysis response: {len(response_text)} characters")
        
        # Parse JSON response
        import json
        import re
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                analysis_data = json.loads(response_text)
            
            # Check if issues were found
            if analysis_data.get('has_issues', False):
                logger.info(f"AI detected logical issues: {analysis_data.get('issue_category', 'UNKNOWN')}")
                
                return ClarificationRequest(
                    agent_name="prompt_interpreter",
                    contradiction_type=analysis_data.get('issue_category', 'GEOMETRY_AMBIGUITY'),
                    detected_issues=analysis_data.get('detected_issues', ['Logical inconsistency detected']),
                    clarification_questions=analysis_data.get('clarification_questions', ['Please clarify the geometry requirements']),
                    suggested_resolutions=analysis_data.get('suggested_resolutions', ['Please provide clearer geometry description']),
                    original_prompt=prompt_text,
                    agent_reasoning=analysis_data.get('reasoning', 'AI detected logical inconsistencies in the geometry prompt')
                )
            else:
                logger.info("AI analysis: No logical issues detected")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI logical analysis JSON: {e}")
            
            # Try to clean the JSON and parse again
            try:
                cleaned_response = _clean_json_response(response_text)
                analysis_data = json.loads(cleaned_response)
                logger.info("Successfully parsed cleaned AI analysis JSON")
                
                # Continue with cleaned analysis data
                if analysis_data.get('has_issues', False):
                    return ClarificationRequest(
                        agent_name="prompt_interpreter",
                        contradiction_type=analysis_data.get('issue_category', 'GEOMETRY_AMBIGUITY'),
                        detected_issues=analysis_data.get('detected_issues', ['Logical inconsistency detected']),
                        clarification_questions=analysis_data.get('clarification_questions', ['Please clarify the geometry requirements']),
                        suggested_resolutions=analysis_data.get('suggested_resolutions', ['Please provide clearer geometry description']),
                        original_prompt=prompt_text,
                        agent_reasoning=analysis_data.get('reasoning', 'AI detected logical inconsistencies in the geometry prompt')
                    )
                else:
                    return None
                    
            except Exception as cleanup_error:
                logger.warning(f"AI analysis JSON cleanup also failed: {cleanup_error}")
            
            # Fall back to simple check if JSON parsing fails
            if any(word in response_text.lower() for word in ['contradiction', 'impossible', 'ambiguous', 'unclear', 'inconsistent']):
                return ClarificationRequest(
                    agent_name="prompt_interpreter",
                    contradiction_type="LOGICAL_INCONSISTENCY",
                    detected_issues=["AI detected potential logical issues in the prompt"],
                    clarification_questions=["Please review and clarify your geometry requirements"],
                    suggested_resolutions=["Provide more specific and consistent geometric descriptions"],
                    original_prompt=prompt_text,
                    agent_reasoning="AI analysis detected potential logical inconsistencies but couldn't parse detailed response"
                )
            return None
            
    except Exception as e:
        logger.error(f"Error in AI logical analysis: {e}")
        return None





def handle(prompt_bundle: PromptBundle) -> Union[GeometrySpec, AgentError, ClarificationRequest]:
    """
    Main entry point for the AI prompt interpreter agent.
    
    Args:
        prompt_bundle: PromptBundle containing user input
        
    Returns:
        GeometrySpec with AI-extracted geometric information, AgentError, or ClarificationRequest for user clarification
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
        
        # AI-powered logical consistency check (unless skipped by user)
        skip_check = getattr(prompt_bundle, 'skip_contradiction_check', False)
        if not skip_check:
            inconsistency_check = _ai_detect_logical_issues(prompt_bundle.text)
            if inconsistency_check:
                return inconsistency_check
        
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