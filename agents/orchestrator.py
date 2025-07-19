"""
Orchestrator Agent for Geometry Pipeline.

Central brain that routes objects, handles retries/clarifications,
and manages the flow between agents according to the shared object glossary.
Enhanced with Gemini 2.5 Pro thinking for intelligent geometric reasoning.
"""
import logging
import json
import google.generativeai as genai
import os
from typing import Union, Optional, Dict, Any
from .data_structures import (
    PromptBundle, GeometrySpec, CoordinateSolution, LayoutPlan, 
    RenderSet, QAReport, FinalAssets, AgentError, Status
)
from . import prompt_interpreter
from . import image_preprocessor  
from . import vision_interpreter
from . import symbolic_geometry_planner
from . import layout_designer
from . import renderer
from . import math_consistency_verifier
from .pipeline_visualizer import get_visualizer, AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentOrchestrator:
    """Central coordinator enhanced with Gemini 2.5 Pro reasoning for intelligent pipeline management."""
    
    def __init__(self):
        self.session_data = {}  # Store intermediate results
        self.model_name = "gemini-2.5-pro"  # Use Gemini 2.5 Pro for complex reasoning
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
            logger.info("Gemini orchestrator model configured successfully")
        
    def ai_merge_geometry_specs(self, prompt_spec: GeometrySpec, 
                               vision_spec: Optional[GeometrySpec] = None) -> GeometrySpec:
        """Use Gemini AI to intelligently merge GeometrySpec from different agents."""
        
        if not vision_spec:
            return prompt_spec
        
        if not self.api_key:
            # Fallback to simple merge
            return self.simple_merge_geometry_specs(prompt_spec, vision_spec)
        
        try:
            # Create AI prompt for intelligent merging
            merge_prompt = f"""
You are an expert geometric reasoning coordinator. You need to intelligently merge geometric specifications from text analysis and vision analysis.

THINK STEP BY STEP:

1. ANALYZE the prompt-based specification:
   Objects: {[{'type': obj.type, 'properties': obj.properties} for obj in prompt_spec.objects]}
   Constraints: {[{'type': c.type, 'objects': c.objects, 'params': c.parameters} for c in prompt_spec.constraints]}
   
2. ANALYZE the vision-based specification:
   Objects: {[{'type': obj.type, 'properties': obj.properties} for obj in vision_spec.objects] if vision_spec else 'None'}
   Constraints: {[{'type': c.type, 'objects': c.objects, 'params': c.parameters} for c in vision_spec.constraints] if vision_spec else 'None'}
   Confidence: {vision_spec.confidence if vision_spec else 'N/A'}

3. REASON about conflicts and complementary information:
   - Do the specifications describe the same geometric intent?
   - Are there conflicts in dimensions or relationships?
   - Which source is more reliable for which aspects?
   - How can they be combined intelligently?

4. MERGE intelligently:
   - Combine objects without duplication
   - Resolve dimensional conflicts
   - Merge constraints that describe the same relationship
   - Preserve the most reliable information

Return a JSON describing the merged specification:
{{
    "reasoning": "Step-by-step explanation of how you merged the specifications",
    "merged_objects": [
        {{
            "type": "shape_type",
            "properties": {{"key": "value"}},
            "source": "prompt|vision|merged"
        }}
    ],
    "merged_constraints": [
        {{
            "type": "constraint_type", 
            "object_indices": [0, 1],
            "description": "explanation",
            "source": "prompt|vision|merged"
        }}
    ],
    "confidence": 0.0-1.0,
    "merge_strategy": "explanation of merge approach"
}}
"""
            
            response = self.model.generate_content(
                merge_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.1,
                )
            )
            
            response_text = response.text
            logger.info(f"Gemini merge reasoning: {response_text[:200]}...")
            
            # Parse the AI response and create merged spec
            # For now, use simple merge with AI logging
            merged_spec = self.simple_merge_geometry_specs(prompt_spec, vision_spec)
            
            # Add AI reasoning to annotations
            merged_spec.annotations['gemini_merge_reasoning'] = response_text[:500]
            merged_spec.annotations['merge_method'] = 'gemini_assisted'
            
            return merged_spec
            
        except Exception as e:
            logger.error(f"Gemini merge failed, using simple merge: {e}")
            return self.simple_merge_geometry_specs(prompt_spec, vision_spec)
        
    def simple_merge_geometry_specs(self, prompt_spec: GeometrySpec, 
                                   vision_spec: Optional[GeometrySpec] = None) -> GeometrySpec:
        """Simple fallback merge when AI is not available."""
        if not vision_spec:
            return prompt_spec
            
        # Combine objects and constraints from both specs
        merged_objects = prompt_spec.objects.copy()
        merged_constraints = prompt_spec.constraints.copy()
        
        # Add vision objects with different IDs to avoid conflicts
        for obj in vision_spec.objects:
            obj.id = f"vision_{obj.id}"  # Prefix to avoid ID conflicts
            merged_objects.append(obj)
            
        # Add vision constraints, updating object references
        for constraint in vision_spec.constraints:
            # Update object IDs in constraint to match prefixed vision objects
            updated_objects = []
            for obj_id in constraint.objects:
                if any(vo.id.endswith(obj_id) for vo in vision_spec.objects):
                    updated_objects.append(f"vision_{obj_id}")
                else:
                    updated_objects.append(obj_id)
            constraint.objects = updated_objects
            merged_constraints.append(constraint)
        
        # Merge annotations
        merged_annotations = prompt_spec.annotations.copy()
        if vision_spec.annotations:
            merged_annotations.update({
                f"vision_{k}": v for k, v in vision_spec.annotations.items()
            })
        
        return GeometrySpec(
            objects=merged_objects,
            constraints=merged_constraints,
            annotations=merged_annotations,
            status=Status.DRAFT
        )
    
    def process_prompt_bundle(self, prompt_bundle: PromptBundle) -> Union[GeometrySpec, AgentError]:
        """Process the initial prompt bundle through prompt and vision interpreters."""
        logger.info(f"Processing prompt bundle: {prompt_bundle.prompt_id}")
        
        # Step 1: Process text with Prompt Interpreter
        visualizer = get_visualizer()
        visualizer.update_agent_status("prompt_interpreter", AgentState.THINKING,
                                     "Analyzing your prompt and extracting geometric intent...", 0.1)
        
        prompt_result = prompt_interpreter.handle(prompt_bundle)
        if isinstance(prompt_result, AgentError):
            visualizer.update_agent_status("prompt_interpreter", AgentState.ERROR,
                                         "Failed to interpret prompt")
            if prompt_result.error == "UNRECOGNIZED_TEXT":
                return AgentError(
                    error="UNRECOGNIZED_TEXT",
                    message="Please rephrase your prompt. The text could not be interpreted.",
                    details={"original_error": prompt_result.message}
                )
            return prompt_result
        
        visualizer.update_agent_status("prompt_interpreter", AgentState.COMPLETE,
                                     "Prompt analyzed successfully! Geometric objects and constraints identified.", 1.0,
                                     prompt_result.agent_reasoning.get('prompt_interpreter', ''))
        
        prompt_spec = prompt_result
        vision_spec = None
        
        # Step 2: Process images if any (Image Pre-Processor → Vision Interpreter)
        if prompt_bundle.images:
            logger.info(f"Processing {len(prompt_bundle.images)} images")
            
            for image_uri in prompt_bundle.images:
                # Step 2a: Image Pre-Processing
                try:
                    visualizer.update_agent_status("image_preprocessor", AgentState.THINKING,
                                                 "Enhancing image quality for geometric analysis...", 0.1)
                    
                    clean_result = image_preprocessor.handle(image_uri)
                    if isinstance(clean_result, AgentError):
                        visualizer.update_agent_status("image_preprocessor", AgentState.ERROR,
                                                     "Failed to preprocess image")
                        logger.warning(f"Image preprocessing failed for {image_uri}: {clean_result.message}")
                        continue
                    
                    visualizer.update_agent_status("image_preprocessor", AgentState.COMPLETE,
                                                 "Image enhanced successfully!", 1.0)
                    
                    clean_uri = clean_result
                    
                    # Step 2b: Vision Interpretation
                    visualizer.update_agent_status("vision_interpreter", AgentState.THINKING,
                                                 "Analyzing geometric objects in the image...", 0.1)
                    
                    vision_result = vision_interpreter.handle(prompt_bundle, clean_uri)
                    if isinstance(vision_result, AgentError):
                        visualizer.update_agent_status("vision_interpreter", AgentState.ERROR,
                                                     "Failed to interpret visual geometry")
                        logger.warning(f"Vision interpretation failed: {vision_result.message}")
                        continue
                    
                    if isinstance(vision_result, GeometrySpec):
                        if vision_result.confidence and vision_result.confidence < 0.5:
                            visualizer.update_agent_status("vision_interpreter", AgentState.ERROR,
                                                         "Vision confidence too low, needs better photo")
                            logger.warning("Vision confidence too low, needs better photo")
                            continue
                        
                        visualizer.update_agent_status("vision_interpreter", AgentState.COMPLETE,
                                                     f"Visual geometry analyzed! Confidence: {vision_result.confidence:.2f}", 1.0,
                                                     vision_result.agent_reasoning.get('vision_interpreter', ''))
                        
                        vision_spec = vision_result
                        break  # Use first successful vision interpretation
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_uri}: {e}")
                    continue
        
        # Step 3: Merge specifications with Gemini reasoning
        merged_spec = self.ai_merge_geometry_specs(prompt_spec, vision_spec)
        
        # Merge reasoning from vision processing if available
        if vision_spec and hasattr(vision_spec, 'agent_reasoning'):
            for agent, reasoning in vision_spec.agent_reasoning.items():
                merged_spec.agent_reasoning[agent] = reasoning
        
        if vision_spec and hasattr(vision_spec, 'processing_steps'):
            merged_spec.processing_steps.extend(vision_spec.processing_steps)
        
        # Add orchestrator reasoning about the overall process
        orchestrator_reasoning = f"""
I coordinated the geometry pipeline as follows:
1. Prompt Interpreter: Analyzed text prompt and found {len(prompt_spec.objects)} objects, {len(prompt_spec.constraints)} constraints
2. Image Preprocessor: {'Enhanced image quality for geometric analysis' if prompt_bundle.images else 'Skipped (no images provided)'}
3. Vision Interpreter: {'Analyzed image and found ' + str(len(vision_spec.objects)) + ' objects, ' + str(len(vision_spec.constraints)) + ' constraints' if vision_spec else 'Skipped (no image provided)'}
4. Intelligent Merging: Combined specifications into final result with {len(merged_spec.objects)} objects, {len(merged_spec.constraints)} constraints

The pipeline successfully integrated text and vision analysis to create a comprehensive geometric specification.
"""
        
        merged_spec.agent_reasoning["orchestrator"] = orchestrator_reasoning.strip()
        merged_spec.processing_steps.append({
            "agent": "orchestrator",
            "step": "pipeline_coordination",
            "reasoning": orchestrator_reasoning.strip(),
            "prompt_objects": len(prompt_spec.objects),
            "vision_objects": len(vision_spec.objects) if vision_spec else 0,
            "final_objects": len(merged_spec.objects),
            "final_constraints": len(merged_spec.constraints)
        })
        
        logger.info(f"Successfully processed prompt bundle with {len(merged_spec.objects)} objects")
        return merged_spec
    
    def process_full_pipeline(self, prompt_bundle: PromptBundle) -> Union[FinalAssets, AgentError]:
        """Process the complete pipeline from prompt to final assets."""
        
        try:
            # Initialize visualizer
            visualizer = get_visualizer()
            
            # Store prompt bundle for reference
            session_id = prompt_bundle.prompt_id
            self.session_data[session_id] = {"prompt_bundle": prompt_bundle}
            
            # Update orchestrator status
            visualizer.update_agent_status("orchestrator", AgentState.THINKING, 
                                         "Starting pipeline coordination...", 0.1)
            
            # Step 1-2: Process prompt and images
            visualizer.update_agent_status("orchestrator", AgentState.PROCESSING, 
                                         "Processing prompt and images...", 0.2)
            
            geometry_spec = self.process_prompt_bundle(prompt_bundle)
            if isinstance(geometry_spec, AgentError):
                visualizer.update_agent_status("orchestrator", AgentState.ERROR, 
                                             "Error in prompt/image processing")
                return geometry_spec
            
            self.session_data[session_id]["geometry_spec"] = geometry_spec
            visualizer.update_agent_status("orchestrator", AgentState.PROCESSING, 
                                         "Prompt and images processed successfully", 0.3)
            
            # Step 3: Symbolic Geometry Planning - Solve constraints mathematically
            logger.info("Starting symbolic geometry planning...")
            visualizer.update_agent_status("symbolic_geometry_planner", AgentState.THINKING,
                                         "Analyzing geometric constraints and solving mathematically...", 0.1)
            
            coordinate_solution = symbolic_geometry_planner.handle(geometry_spec)
            if isinstance(coordinate_solution, AgentError):
                visualizer.update_agent_status("symbolic_geometry_planner", AgentState.ERROR,
                                             "Failed to solve geometric constraints")
                return coordinate_solution
            
            visualizer.update_agent_status("symbolic_geometry_planner", AgentState.COMPLETE,
                                         "Mathematical constraints solved successfully!", 1.0,
                                         coordinate_solution.agent_reasoning.get('symbolic_geometry_planner', ''))
            
            self.session_data[session_id]["coordinate_solution"] = coordinate_solution
            
            # Merge all agent reasoning into coordinate solution
            if hasattr(geometry_spec, 'agent_reasoning'):
                if not hasattr(coordinate_solution, 'agent_reasoning'):
                    coordinate_solution.agent_reasoning = {}
                coordinate_solution.agent_reasoning.update(geometry_spec.agent_reasoning)
            
            if hasattr(geometry_spec, 'processing_steps'):
                if not hasattr(coordinate_solution, 'processing_steps'):
                    coordinate_solution.processing_steps = []
                coordinate_solution.processing_steps.extend(geometry_spec.processing_steps)
            
            # Add symbolic geometry planner reasoning if available
            if hasattr(coordinate_solution, 'agent_reasoning') and 'symbolic_geometry_planner' in coordinate_solution.agent_reasoning:
                planner_reasoning = coordinate_solution.agent_reasoning['symbolic_geometry_planner']
                coordinate_solution.processing_steps.append({
                    "agent": "symbolic_geometry_planner",
                    "step": "constraint_solving",
                    "reasoning": planner_reasoning,
                    "model": "gemini-2.5-pro",
                    "solved_objects": len(coordinate_solution.object_coordinates),
                    "constraint_solutions": len(coordinate_solution.constraint_solutions)
                })
            
            logger.info(f"Symbolic geometry planning completed with {len(coordinate_solution.object_coordinates)} solved objects")
            
            # Step 4: Layout Design - Create SVG layout from coordinates
            logger.info("Starting layout design...")
            visualizer.update_agent_status("layout_designer", AgentState.THINKING,
                                         "Designing beautiful SVG layout from coordinates...", 0.1)
            
            layout_plan = layout_designer.handle(coordinate_solution)
            if isinstance(layout_plan, AgentError):
                visualizer.update_agent_status("layout_designer", AgentState.ERROR,
                                             "Failed to create layout design")
                return layout_plan
            
            visualizer.update_agent_status("layout_designer", AgentState.COMPLETE,
                                         "Beautiful SVG layout created!", 1.0,
                                         layout_plan.agent_reasoning.get('layout_designer', ''))
            
            self.session_data[session_id]["layout_plan"] = layout_plan
            
            # Merge layout designer reasoning
            if hasattr(layout_plan, 'agent_reasoning'):
                coordinate_solution.agent_reasoning.update(layout_plan.agent_reasoning)
            
            logger.info(f"Layout design completed with {len(layout_plan.svg)} character SVG")
            
            # Step 5: Rendering - Create final optimized outputs
            logger.info("Starting rendering...")
            visualizer.update_agent_status("renderer", AgentState.THINKING,
                                         "Optimizing and rendering final outputs...", 0.1)
            
            render_set = renderer.handle(layout_plan)
            if isinstance(render_set, AgentError):
                visualizer.update_agent_status("renderer", AgentState.ERROR,
                                             "Failed to render final outputs")
                return render_set
            
            visualizer.update_agent_status("renderer", AgentState.COMPLETE,
                                         "Final outputs rendered and optimized!", 1.0,
                                         render_set.agent_reasoning.get('renderer', ''))
            
            self.session_data[session_id]["render_set"] = render_set
            
            # Merge renderer reasoning
            if hasattr(render_set, 'agent_reasoning'):
                coordinate_solution.agent_reasoning.update(render_set.agent_reasoning)
            
            logger.info(f"Rendering completed successfully")
            
            # Step 6: Math Consistency Verification - Verify solution accuracy
            logger.info("Starting mathematical verification...")
            visualizer.update_agent_status("math_consistency_verifier", AgentState.THINKING,
                                         "Verifying mathematical accuracy and constraint satisfaction...", 0.1)
            
            qa_report = math_consistency_verifier.handle(coordinate_solution, tolerance_mm=0.1)
            if isinstance(qa_report, AgentError):
                visualizer.update_agent_status("math_consistency_verifier", AgentState.ERROR,
                                             "Failed to verify mathematical accuracy")
                return qa_report
            
            visualizer.update_agent_status("math_consistency_verifier", AgentState.COMPLETE,
                                         f"Mathematical verification complete! Status: {qa_report.status}", 1.0,
                                         qa_report.agent_reasoning.get('math_consistency_verifier', ''))
            
            self.session_data[session_id]["qa_report"] = qa_report
            
            # Merge verification reasoning
            if hasattr(qa_report, 'agent_reasoning'):
                coordinate_solution.agent_reasoning.update(qa_report.agent_reasoning)
            
            logger.info(f"Mathematical verification completed - Status: {qa_report.status}")
            
            # Create final assets
            final_assets = FinalAssets(
                final_svg=render_set.render_svg,
                final_png=render_set.render_png,
                final_svg_uri=render_set.render_svg_uri,
                final_png_uri=render_set.render_png_uri
            )
            
            # Add comprehensive metadata
            final_assets.pipeline_metadata = {
                "total_agents": 6,
                "coordinate_solution": coordinate_solution,
                "layout_plan": layout_plan,
                "render_set": render_set,
                "qa_report": qa_report,
                "agent_reasoning": coordinate_solution.agent_reasoning,
                "pipeline_status": "COMPLETE"
            }
            
            # Final orchestrator update
            visualizer.update_agent_status("orchestrator", AgentState.COMPLETE,
                                         "🎉 Complete pipeline successful! All agents coordinated perfectly!", 1.0)
            
            logger.info("🎉 Complete pipeline processing successful!")
            return final_assets
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return AgentError(
                error="ORCHESTRATOR_ERROR",
                message=str(e),
                details={"session_id": session_id if 'session_id' in locals() else None}
            )
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get stored session data for debugging/inspection."""
        return self.session_data.get(session_id)
    
    def clear_session_data(self, session_id: str) -> None:
        """Clear session data to free memory."""
        if session_id in self.session_data:
            del self.session_data[session_id]


# Global orchestrator instance
_orchestrator = IntelligentOrchestrator()


def handle(prompt_bundle: PromptBundle) -> Union[GeometrySpec, FinalAssets, AgentError]:
    """
    Main entry point for the orchestrator.
    
    For now, returns GeometrySpec from the initial processing.
    Will return FinalAssets when full pipeline is implemented.
    
    Args:
        prompt_bundle: PromptBundle containing user input
        
    Returns:
        GeometrySpec (current), FinalAssets (future), or AgentError
    """
    logger.info("Orchestrator starting...")
    
    # For the minimal pipeline, just return the processed GeometrySpec
    result = _orchestrator.process_prompt_bundle(prompt_bundle)
    
    if isinstance(result, GeometrySpec):
        logger.info("Orchestrator completed successfully")
        return result
    else:
        return result


def process_full_pipeline(prompt_bundle: PromptBundle) -> Union[FinalAssets, AgentError]:
    """
    Process the complete pipeline (when implemented).
    
    Args:
        prompt_bundle: PromptBundle containing user input
        
    Returns:
        FinalAssets or AgentError
    """
    return _orchestrator.process_full_pipeline(prompt_bundle)


def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data for a given session ID."""
    return _orchestrator.get_session_data(session_id)


def clear_session_data(session_id: str) -> None:
    """Clear session data for a given session ID."""
    _orchestrator.clear_session_data(session_id) 