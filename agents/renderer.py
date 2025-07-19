#!/usr/bin/env python3
"""
Renderer Agent - Creates final rendered outputs from layout plans
Uses Gemini 2.5 Pro for intelligent rendering decisions and optimizations
"""

import os
import logging
import json
import re
import tempfile
import base64
from typing import Dict, Any, Optional
import google.generativeai as genai
from datetime import datetime

from .data_structures import LayoutPlan, RenderSet, AgentError, Status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Renderer:
    """AI-powered renderer using Gemini 2.5 Pro for intelligent output generation."""
    
    def __init__(self):
        """Initialize the Renderer with Gemini 2.5 Pro."""
        self.model = None
        self._setup_gemini()
        
        # Create debug directory for JSON logs
        self.debug_dir = "debug_json_responses"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def _save_debug_json(self, response_text: str, prompt_id: str, status: str):
        """Save raw JSON response for debugging purposes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"renderer_{prompt_id[:8]}_{timestamp}_{status}.json"
            filepath = os.path.join(self.debug_dir, filename)
            
            debug_data = {
                "agent": "renderer",
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
        """Configure Gemini 2.5 Pro for rendering optimization."""
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
            logger.info("Gemini renderer configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            self.model = None

    def handle(self, layout_plan: LayoutPlan) -> RenderSet:
        """
        Create final rendered outputs from layout plan.
        
        Args:
            layout_plan: SVG layout plan with styling information
            
        Returns:
            RenderSet with final rendered files
        """
        try:
            logger.info("Renderer agent starting...")
            
            if not self.model:
                return AgentError(
                    error="MODEL_UNAVAILABLE",
                    message="Gemini model not available",
                    details={"agent": "renderer", "error": "Model initialization failed"}
                )
            
            # Create intelligent rendered output using Gemini 2.5 Pro
            render_set = self._render_with_ai(layout_plan)
            
            if isinstance(render_set, AgentError):
                return render_set
            
            logger.info(f"Renderer completed successfully")
            return render_set
            
        except Exception as e:
            logger.error(f"Error in renderer: {e}")
            return AgentError(
                error="RENDERING_FAILED",
                message=f"Rendering failed: {str(e)}",
                details={"agent": "renderer", "error": str(e)}
            )

    def _render_with_ai(self, layout_plan: LayoutPlan) -> RenderSet:
        """Use Gemini 2.5 Pro to optimize and enhance the rendered output."""
        
        # Prepare layout data for AI analysis
        render_prompt = f"""
You are an expert SVG renderer and optimizer. Enhance and optimize this SVG layout for final rendering.

LAYOUT PLAN:
SVG Content: {layout_plan.svg[:1000]}...
Style Tokens: {json.dumps(layout_plan.style_tokens, indent=2)}
Labels: {json.dumps(layout_plan.labels, indent=2)}

REQUIREMENTS:
1. Optimize the SVG for clarity, precision, and visual appeal
2. Ensure proper scaling and viewBox settings
3. Add professional styling and consistent colors
4. Include clear labels and annotations
5. Optimize for both web display and potential printing
6. Add accessibility features (titles, descriptions)
7. Ensure mathematical accuracy is preserved

RESPONSE FORMAT:
```json
{{
    "reasoning": "Your step-by-step thinking about rendering optimizations...",
    "optimized_svg": "Complete optimized SVG markup...",
    "rendering_decisions": {{
        "optimization_type": "performance, clarity, or aesthetic",
        "styling_enhancements": "description of styling improvements",
        "accessibility_features": "added accessibility elements",
        "quality_improvements": "specific quality enhancements"
    }},
    "metadata": {{
        "svg_size": "estimated file size",
        "complexity_score": "1-10 rendering complexity",
        "recommended_format": "svg, png, or both"
    }}
}}
```

Focus on creating a professional, publication-ready geometric diagram.
"""
        
        try:
            logger.info(f"Optimizing SVG rendering with AI")
            
            response = self.model.generate_content(render_prompt)
            response_text = response.text
            
            # Save raw response for debugging
            layout_id = getattr(layout_plan, 'layout_id', 'rendering')
            self._save_debug_json(response_text, layout_id, "raw_response")
            
            logger.info(f"Gemini rendering response: {len(response_text)} characters")
            
            # Extract reasoning for frontend display
            reasoning_match = re.search(r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"', response_text, re.DOTALL)
            reasoning = reasoning_match.group(1) if reasoning_match else "Rendering optimization completed with AI enhancements."
            
            # Store reasoning (attach to layout_plan for pipeline tracking)
            if not hasattr(layout_plan, 'agent_reasoning'):
                layout_plan.agent_reasoning = {}
            layout_plan.agent_reasoning['renderer'] = reasoning
            
            logger.info(f"Gemini reasoning extracted: {reasoning[:100]}...")
            
            # Parse JSON response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                render_data = json.loads(json_str)
                # Save successfully parsed JSON
                self._save_debug_json(json_str, layout_id, "parsed_success")
            else:
                # Fallback parsing
                render_data = json.loads(response_text)
                # Save fallback parsed JSON
                self._save_debug_json(response_text, layout_id, "fallback_parsed")
            
            # Extract optimized SVG
            optimized_svg = render_data.get('optimized_svg', layout_plan.svg)
            rendering_decisions = render_data.get('rendering_decisions', {})
            metadata = render_data.get('metadata', {})
            
            # Save SVG to temporary file
            svg_file = self._save_svg_file(optimized_svg)
            
            # Create render set
            render_set = RenderSet(
                render_svg=optimized_svg,
                render_svg_uri=svg_file,
                render_png=None,  # PNG generation would require additional tools
                render_png_uri=None
            )
            
            # Add rendering metadata
            render_set.rendering_decisions = rendering_decisions
            render_set.metadata = metadata
            render_set.agent_reasoning = getattr(layout_plan, 'agent_reasoning', {})
            
            return render_set
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            
            # Save failed JSON for debugging
            self._save_debug_json(response_text, layout_id, "parse_failed")
            
            # Try to clean the JSON and parse again
            try:
                cleaned_response = self._clean_json_response(response_text)
                render_decisions = json.loads(cleaned_response)
                
                # Save cleaned successful JSON
                self._save_debug_json(cleaned_response, layout_id, "cleaned_success")
                
                # Continue with cleaned JSON
                final_svg = render_decisions.get('optimized_svg', '')
                if final_svg:
                    logger.info("Successfully parsed cleaned JSON response")
                    return RenderSet(
                        render_svg=final_svg,
                        rendering_decisions=render_decisions.get('optimization_decisions', {}),
                        metadata=render_decisions.get('metadata', {}),
                        agent_reasoning={'renderer': reasoning}
                    )
                    
            except Exception as cleanup_error:
                logger.warning(f"JSON cleanup also failed: {cleanup_error}")
                # Save final failed attempt
                self._save_debug_json(cleaned_response if 'cleaned_response' in locals() else response_text, 
                                    layout_id, "cleanup_failed")
            
            return self._create_fallback_render(layout_plan)
            
        except Exception as e:
            logger.error(f"Gemini rendering failed: {e}")
            return self._create_fallback_render(layout_plan)

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
        cleaned = re.sub(r'}(\s*\n\s*)\{', r'},\1{', cleaned)
        cleaned = re.sub(r'}(\s+)\{', r'},\1{', cleaned)
        
        # Fix 2: Enhanced SVG quote escaping for complex content
        def fix_svg_field_quotes(text, field_name):
            # Enhanced parsing to handle mixed escaping scenarios
            # 1. Find SVG field content precisely
            # 2. Count consecutive backslashes to determine escape state
            # 3. Normalize all quotes: remove existing escaping
            # 4. Properly re-escape all quotes for JSON consistency
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
        
        # Stage 4: Test if fixes worked
        try:
            json.loads(cleaned)
            return cleaned.strip()
        except:
            # Stage 5: More aggressive cleaning for complex cases
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
    
    def _create_fallback_render(self, layout_plan: LayoutPlan) -> RenderSet:
        """Create a simple fallback render when AI fails."""
        logger.warning("Creating fallback render")
        
        # Basic SVG optimization
        optimized_svg = self._basic_svg_optimization(layout_plan.svg)
        svg_file = self._save_svg_file(optimized_svg)
        
        render_set = RenderSet(
            render_svg=optimized_svg,
            render_svg_uri=svg_file,
            render_png=None,
            render_png_uri=None
        )
        
        render_set.rendering_decisions = {
            "optimization_type": "basic",
            "styling_enhancements": "minimal cleanup applied",
            "quality_improvements": "basic fallback rendering"
        }
        
        return render_set

    def _basic_svg_optimization(self, svg_content: str) -> str:
        """Apply basic SVG optimizations."""
        # Add basic optimizations
        if 'xmlns=' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
        
        # Ensure proper structure
        if not svg_content.strip().startswith('<svg'):
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">{svg_content}</svg>'
        
        return svg_content

    def _save_svg_file(self, svg_content: str) -> str:
        """Save SVG content to a temporary file and return the path."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(svg_content)
                svg_path = f.name
            
            logger.info(f"SVG saved to: {svg_path}")
            return svg_path
            
        except Exception as e:
            logger.error(f"Failed to save SVG file: {e}")
            return ""


# Global instance
renderer = Renderer()

def handle(layout_plan: LayoutPlan) -> RenderSet:
    """Handle rendering request."""
    return renderer.handle(layout_plan) 