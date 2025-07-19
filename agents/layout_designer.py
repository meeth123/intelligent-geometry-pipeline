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
            self.model = genai.GenerativeModel('gemini-2.5-flash')
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
                layout_data = json.loads(json_match.group(1))
            else:
                # Fallback parsing
                layout_data = json.loads(response_text)
            
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
                style_tokens=style_tokens
            )
            
            return layout_plan
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return self._create_fallback_layout(coordinate_solution)
            
        except Exception as e:
            logger.error(f"Gemini layout design failed: {e}")
            return self._create_fallback_layout(coordinate_solution)

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
            }
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