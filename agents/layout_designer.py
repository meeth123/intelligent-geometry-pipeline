#!/usr/bin/env python3
"""
Layout Designer Agent - Creates intelligent SVG layouts from coordinate solutions
Uses Gemini 2.5 Pro for optimal layout design and styling decisions
"""

import os
import logging
import json
import re
from typing import Dict, Any, Optional
import google.generativeai as genai
from datetime import datetime

from .data_structures import CoordinateSolution, LayoutPlan, AgentError, Status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayoutDesigner:
    """AI-powered layout designer using Gemini 2.5 Pro for intelligent SVG creation."""
    
    def __init__(self):
        """Initialize the Layout Designer with Gemini 2.5 Pro."""
        self.model = None
        self._setup_gemini()
        
        # Create debug directory for JSON logs
        self.debug_dir = "debug_json_responses"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def _save_debug_json(self, response_text: str, prompt_id: str, status: str):
        """Save raw JSON response for debugging purposes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"layout_designer_{prompt_id[:8]}_{timestamp}_{status}.json"
            filepath = os.path.join(self.debug_dir, filename)
            
            debug_data = {
                "agent": "layout_designer",
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
    
    def _setup_gemini(self):
        """Configure Gemini 2.5 Pro for layout design."""
        try:
            # Try to get API key from environment, with fallback
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                # Fallback API key setting
                api_key = "AIzaSyAHCbEYuASKZkr2adn2-CanH0aF7vusnus"
                os.environ['GOOGLE_API_KEY'] = api_key
                logger.warning("Setting Google API key directly in environment")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini layout designer configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            self.model = None

    def handle(self, coordinate_solution: CoordinateSolution) -> LayoutPlan:
        """
        Create an intelligent SVG layout from coordinate solution.
        
        Args:
            coordinate_solution: Solved geometric constraints with coordinates
            
        Returns:
            LayoutPlan with SVG layout and styling information
        """
        try:
            logger.info("Layout Designer agent starting...")
            
            if not self.model:
                return AgentError(
                    error="MODEL_UNAVAILABLE",
                    message="Gemini model not available",
                    details={"agent": "layout_designer", "error": "Model initialization failed"}
                )
            
            # Create intelligent layout using Gemini 2.5 Pro
            layout_plan = self._design_layout_with_ai(coordinate_solution)
            
            if isinstance(layout_plan, AgentError):
                return layout_plan
            
            logger.info(f"Layout Designer completed successfully with {len(layout_plan.svg)} character SVG")
            return layout_plan
            
        except Exception as e:
            logger.error(f"Error in layout designer: {e}")
            return AgentError(
                error="LAYOUT_DESIGN_FAILED",
                message=f"Layout design failed: {str(e)}",
                details={"agent": "layout_designer", "error": str(e)}
            )

    def _design_layout_with_ai(self, coordinate_solution: CoordinateSolution) -> LayoutPlan:
        """Use Gemini 2.5 Pro to create an intelligent SVG layout."""
        
        # Prepare coordinate data for AI analysis
        layout_prompt = f"""
You are an expert SVG layout designer. Create a beautiful, precise SVG layout from this geometric coordinate solution.

COORDINATE SOLUTION:
Objects: {json.dumps(coordinate_solution.object_coordinates, indent=2)}
Constraints: {json.dumps(coordinate_solution.constraint_solutions, indent=2)}
Coordinate System: {json.dumps(coordinate_solution.coordinate_system, indent=2)}

REQUIREMENTS:
1. Create a complete SVG with viewBox, proper scaling, and clean geometry
2. Use appropriate stroke widths, colors, and styling for clarity
3. Include proper labels for key points and measurements
4. Ensure mathematical precision in coordinate placement
5. Make it visually appealing with good contrast and spacing
6. Add grid lines or axes if helpful for understanding

RESPONSE FORMAT:
```json
{{
    "reasoning": "Your step-by-step thinking about the optimal layout design...",
    "svg_content": "Complete SVG markup string...",
    "layout_decisions": {{
        "viewbox": "chosen viewBox dimensions and reasoning",
        "scaling": "scaling factor and approach",
        "styling": "color scheme and styling choices",
        "labels": "labeling strategy and placement"
    }},
    "style_tokens": {{
        "primary_color": "#color",
        "secondary_color": "#color",
        "stroke_width": "value",
        "font_family": "font choice",
        "grid_enabled": true/false
    }}
}}
```

Think step-by-step about the best way to visualize this geometry clearly and beautifully.
"""
        
        try:
            logger.info(f"Designing layout for {len(coordinate_solution.object_coordinates)} objects")
            
            response = self.model.generate_content(layout_prompt)
            response_text = response.text
            
            # Save raw response for debugging
            coord_id = getattr(coordinate_solution, 'solution_id', 'layout_design')
            self._save_debug_json(response_text, coord_id, "raw_response")
            
            logger.info(f"Gemini layout design response: {len(response_text)} characters")
            
            # Extract reasoning for frontend display
            reasoning_match = re.search(r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"', response_text, re.DOTALL)
            reasoning = reasoning_match.group(1) if reasoning_match else "Layout design completed with AI optimization."
            
            # Store reasoning for frontend
            if not hasattr(coordinate_solution, 'agent_reasoning'):
                coordinate_solution.agent_reasoning = {}
            coordinate_solution.agent_reasoning['layout_designer'] = reasoning
            
            logger.info(f"Gemini reasoning extracted: {reasoning[:100]}...")
            
            # Parse JSON response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                layout_data = json.loads(json_str)
                # Save successfully parsed JSON
                self._save_debug_json(json_str, coord_id, "parsed_success")
            else:
                # Fallback parsing
                layout_data = json.loads(response_text)
                # Save fallback parsed JSON
                self._save_debug_json(response_text, coord_id, "fallback_parsed")
            
            # Extract layout components
            svg_content = layout_data.get('svg_content', self._create_fallback_svg(coordinate_solution))
            style_tokens = layout_data.get('style_tokens', {})
            layout_decisions = layout_data.get('layout_decisions', {})
            
            # Create labels from layout decisions
            labels = []
            if 'labels' in layout_decisions:
                labels.append({
                    "type": "layout_info",
                    "content": layout_decisions['labels'],
                    "position": "metadata"
                })
            
            # Create layout plan
            layout_plan = LayoutPlan(
                svg=svg_content,
                labels=labels,
                style_tokens=style_tokens,
                agent_reasoning={'layout_designer': reasoning}
            )
            
            return layout_plan
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            
            # Save failed JSON for debugging
            self._save_debug_json(response_text, coord_id, "parse_failed")
            
            # Try to clean the JSON and parse again
            try:
                # Remove control characters and fix common JSON issues
                cleaned_response = self._clean_json_response(response_text)
                layout_decisions = json.loads(cleaned_response)
                
                # Save cleaned successful JSON
                self._save_debug_json(cleaned_response, coord_id, "cleaned_success")
                
                # Continue with cleaned JSON
                svg_content = layout_decisions.get('svg_content', '')
                labels = []
                style_tokens = layout_decisions.get('style_tokens', {})
                
                if svg_content:
                    logger.info("Successfully parsed cleaned JSON response")
                    layout_plan = LayoutPlan(
                        svg=svg_content,
                        labels=labels,
                        style_tokens=style_tokens,
                        agent_reasoning={'layout_designer': reasoning}
                    )
                    return layout_plan
                    
            except Exception as cleanup_error:
                logger.warning(f"JSON cleanup also failed: {cleanup_error}")
                # Save final failed attempt
                self._save_debug_json(cleaned_response if 'cleaned_response' in locals() else response_text, 
                                    coord_id, "cleanup_failed")
            
            return self._create_fallback_layout(coordinate_solution)
            
        except Exception as e:
            logger.error(f"Gemini layout design failed: {e}")
            return self._create_fallback_layout(coordinate_solution)

    def _clean_json_response(self, response_text: str) -> str:
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
        
        # Fix 1: Missing commas between array objects
        cleaned = re.sub(r'}(\s*\n\s*)\{', r'},\1{', cleaned)  # Handle newlines
        cleaned = re.sub(r'}(\s+)\{', r'},\1{', cleaned)  # Handle spaces/tabs
        
        # More aggressive comma fixing
        lines = cleaned.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped == '}' and i + 1 < len(lines):
                next_line_idx = i + 1
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    next_line_idx += 1
                
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx].strip()
                    if next_line == '{':
                        fixed_lines.append(line.rstrip() + ',')
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        cleaned = '\n'.join(fixed_lines)
        
        # Fix 2: Enhanced SVG quote escaping for complex content
        def fix_svg_field_quotes(text, field_name):
            field_pattern = f'"{field_name}"\\s*:\\s*"'
            match = re.search(field_pattern, text)
            if not match:
                return text
            
            content_start = match.end() - 1  # Position of opening quote
            
            # Enhanced parsing to handle mixed escaping scenarios
            i = content_start + 1  # Start after opening quote
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
                            remaining = text[i+1:].strip()
                            if remaining.startswith(',') or remaining.startswith('}'):
                                content_end = i
                                break
                            i += 1
                            continue
                    continue
                    
                elif char == '"':
                    # Direct unescaped quote - check if it ends the field
                    remaining = text[i+1:].strip()
                    if remaining.startswith(',') or remaining.startswith('}'):
                        content_end = i
                        break
                    i += 1
                    continue
                else:
                    i += 1
            
            if content_end > content_start + 1:
                # Extract the SVG content (between quotes)
                svg_content = text[content_start + 1:content_end]
                
                # Normalize quotes: remove all existing escaping, then properly escape
                normalized = svg_content
                # Remove double escaping first (\\\\" -> \\")
                normalized = normalized.replace('\\\\"', '\\"')
                # Remove single escaping (\\") -> ")
                normalized = normalized.replace('\\"', '"')
                # Now properly escape all quotes for JSON
                fixed_content = normalized.replace('"', '\\"')
                
                return text[:content_start + 1] + fixed_content + text[content_end:]
            
            return text
        
        for svg_field in ['optimized_svg', 'svg_content', 'render_svg']:
            cleaned = fix_svg_field_quotes(cleaned, svg_field)
        
        # Fix 3: Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix 4: Corrupted HTML/SVG closing tags (Issue #3)
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
        
        # Stage 4: Fix structural completeness issues
        open_braces = cleaned.count('{')
        close_braces = cleaned.count('}')
        open_brackets = cleaned.count('[')
        close_brackets = cleaned.count(']')
        
        missing_braces = open_braces - close_braces
        missing_brackets = open_brackets - close_brackets
        
        if missing_braces > 0 or missing_brackets > 0:
            for _ in range(missing_brackets):
                cleaned += '\n    ]'
            for _ in range(missing_braces):
                cleaned += '\n}'
        
        # Check for unclosed strings
        quote_count = 0
        i = 0
        while i < len(cleaned):
            if cleaned[i] == '\\':
                i += 2
                continue
            elif cleaned[i] == '"':
                quote_count += 1
            i += 1
        
        if quote_count % 2 == 1:
            cleaned += '"'
        
        # Stage 5: Test if fixes worked
        try:
            json.loads(cleaned)
            return cleaned.strip()
        except:
            # Stage 6: More aggressive cleaning for complex cases
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                extracted = json_match.group()
                
                extracted = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', extracted)
                extracted = re.sub(r',(\s*[}\]])', r'\1', extracted)
                
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
                        result += '\\n'
                    elif char == '\t' and in_string:
                        result += '\\t'
                    elif char == '\r' and in_string:
                        result += '\\r'
                    else:
                        result += char
                
                return result.strip()
        
        return cleaned.strip()
    
    def _create_fallback_layout(self, coordinate_solution: CoordinateSolution) -> LayoutPlan:
        """Create a simple fallback layout when AI fails."""
        logger.warning("Creating fallback layout")
        
        svg_content = self._create_fallback_svg(coordinate_solution)
        
        return LayoutPlan(
            svg=svg_content,
            labels=[{"type": "fallback", "content": "Basic layout generated", "position": "top"}],
            style_tokens={
                "primary_color": "#2E86AB",
                "secondary_color": "#A23B72", 
                "stroke_width": "2",
                "font_family": "Arial, sans-serif"
            },
            agent_reasoning={'layout_designer': "Fallback layout created due to AI parsing error. Using simple geometric rendering."}
        )

    def _create_fallback_svg(self, coordinate_solution: CoordinateSolution) -> str:
        """Create a basic SVG from coordinate solution."""
        svg_parts = [
            '<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">',
            '<style>.object { stroke: #2E86AB; stroke-width: 2; fill: none; }</style>',
            '<rect x="0" y="0" width="400" height="400" fill="#f9f9f9" stroke="#ddd"/>',
        ]
        
        # Add basic geometric objects
        y_offset = 50
        for obj_id, coords in coordinate_solution.object_coordinates.items():
            if isinstance(coords, dict):
                if 'center' in coords and 'radius' in coords:
                    # Circle
                    cx, cy = coords['center'] if isinstance(coords['center'], (list, tuple)) else (200, y_offset)
                    r = coords['radius']
                    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r*10}" class="object"/>')
                elif 'points' in coords:
                    # Polygon/Line
                    points_str = ' '.join([f"{p[0]*10+200},{p[1]*10+y_offset}" for p in coords['points']])
                    svg_parts.append(f'<polyline points="{points_str}" class="object"/>')
            y_offset += 100
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)


# Global instance
layout_designer = LayoutDesigner()

def handle(coordinate_solution: CoordinateSolution) -> LayoutPlan:
    """Handle layout design request."""
    return layout_designer.handle(coordinate_solution) 