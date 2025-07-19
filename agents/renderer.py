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
            self.model = genai.GenerativeModel('gemini-2.5-flash')
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
                render_data = json.loads(json_match.group(1))
            else:
                # Fallback parsing
                render_data = json.loads(response_text)
            
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
            return self._create_fallback_render(layout_plan)
            
        except Exception as e:
            logger.error(f"Gemini rendering failed: {e}")
            return self._create_fallback_render(layout_plan)

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