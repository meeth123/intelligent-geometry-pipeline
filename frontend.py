#!/usr/bin/env python3
"""
Streamlit Frontend for AI-Powered Geometry Pipeline
Enhanced with O3 Thinking Process Visualization
"""
import streamlit as st
import json
import tempfile
import os
from agents.data_structures import PromptBundle, Status
from agents.orchestrator import handle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_thinking_expander(agent_name: str, reasoning: str, step_data: dict = None):
    """Create an expandable section showing Gemini thinking process."""
    
    # Color coding for different agents
    colors = {
        "prompt_interpreter": "🧠",
        "vision_interpreter": "👁️", 
        "image_preprocessor": "🖼️",
        "symbolic_geometry_planner": "🔢",
        "layout_designer": "🎨",
        "renderer": "🖨️",
        "math_consistency_verifier": "✅",
        "orchestrator": "🎯"
    }
    
    icon = colors.get(agent_name, "🤖")
    
    with st.expander(f"{icon} {agent_name.replace('_', ' ').title()} - Gemini Thinking Process"):
        # Show the reasoning
        st.markdown("**🧠 Gemini Reasoning:**")
        st.text_area(
            label="Gemini Reasoning",
            value=reasoning,
            height=200,
            disabled=True,
            key=f"reasoning_{agent_name}_{hash(reasoning[:50])}",
            label_visibility="collapsed"
        )
        
        # Show additional step data if available
        if step_data:
            st.markdown("**📊 Processing Details:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "model" in step_data:
                    st.metric("AI Model", step_data["model"])
                if "objects_found" in step_data:
                    st.metric("Objects Found", step_data["objects_found"])
                if "confidence" in step_data:
                    st.metric("Confidence", f"{step_data['confidence']:.2f}")
                    
            with col2:
                if "constraints_found" in step_data:
                    st.metric("Constraints Found", step_data["constraints_found"])
                if "objects_detected" in step_data:
                    st.metric("Objects Detected", step_data["objects_detected"])
                if "constraints_detected" in step_data:
                    st.metric("Constraints Detected", step_data["constraints_detected"])

def display_pipeline_flow(geometry_spec):
    """Display the complete Gemini pipeline flow with thinking processes."""
    
    st.markdown("## 🧠 Gemini AI Pipeline - Thinking Process")
    st.markdown("See how each Gemini agent reasoned through the geometric problem:")
    
    # Show processing steps in order
    if hasattr(geometry_spec, 'processing_steps') and geometry_spec.processing_steps:
        for i, step in enumerate(geometry_spec.processing_steps):
            agent = step.get("agent", "unknown")
            reasoning = step.get("reasoning", "No reasoning provided")
            
            # Create thinking expander for each step with unique identifier
            create_thinking_expander(f"{agent}_step_{i}", reasoning, step)
    
    # Show overall agent reasoning
    if hasattr(geometry_spec, 'agent_reasoning') and geometry_spec.agent_reasoning:
        st.markdown("### 💭 Agent Reasoning Summary")
        
        for i, (agent, reasoning) in enumerate(geometry_spec.agent_reasoning.items()):
            if reasoning and reasoning.strip():
                create_thinking_expander(f"{agent}_summary_{i}", reasoning)

def display_qa_report(qa_report):
    """Display QA report results from math consistency verifier."""
    
    st.markdown("## ✅ Mathematical Verification Report")
    
    # Status and metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_icon = "✅" if qa_report.status.value == "VERIFIED" else "⚠️" if qa_report.status.value == "DRAFT" else "❌"
        st.metric("Verification Status", f"{status_icon} {qa_report.status.value}")
    with col2:
        st.metric("Tolerance", f"{qa_report.tolerance_mm}mm")
    with col3:
        issue_count = len(qa_report.issues)
        st.metric("Issues Found", issue_count)
    
    # Verification details
    if hasattr(qa_report, 'verification_data') and qa_report.verification_data:
        verification_data = qa_report.verification_data
        
        if 'constraint_verification' in verification_data:
            st.markdown("### 📏 Constraint Verification")
            constraint_data = verification_data['constraint_verification']
            
            col1, col2 = st.columns(2)
            with col1:
                total = constraint_data.get('total_constraints', 0)
                satisfied = constraint_data.get('satisfied_constraints', 0)
                st.metric("Constraints Satisfied", f"{satisfied}/{total}")
            with col2:
                if total > 0:
                    satisfaction_rate = satisfied / total
                    st.metric("Satisfaction Rate", f"{satisfaction_rate:.1%}")
        
        if 'overall_quality_score' in verification_data:
            st.markdown("### 📊 Quality Score")
            quality_score = verification_data['overall_quality_score']
            st.progress(quality_score)
            st.write(f"Overall Quality: {quality_score:.2f}/1.0")
    
    # Issues
    if qa_report.issues:
        st.markdown("### ⚠️ Issues Identified")
        for issue in qa_report.issues:
            if issue.startswith('[CRITICAL]'):
                st.error(issue)
            elif issue.startswith('[WARNING]'):
                st.warning(issue)
            else:
                st.info(issue)

def display_layout_and_rendering_results(layout_plan, render_set):
    """Display layout design and rendering results."""
    
    st.markdown("## 🎨 Layout Design & Rendering")
    
    # Layout information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📐 Layout Design")
        st.metric("SVG Size", f"{len(layout_plan.svg)} characters")
        st.metric("Style Tokens", len(layout_plan.style_tokens))
        st.metric("Labels", len(layout_plan.labels))
        
        if layout_plan.style_tokens:
            with st.expander("🎨 Style Configuration"):
                st.json(layout_plan.style_tokens)
    
    with col2:
        st.markdown("### 🖨️ Rendering")
        if hasattr(render_set, 'rendering_decisions'):
            decisions = render_set.rendering_decisions
            optimization_type = decisions.get('optimization_type', 'unknown')
            st.metric("Optimization Type", optimization_type.title())
            
            if 'quality_improvements' in decisions:
                st.write(f"**Quality Improvements:** {decisions['quality_improvements']}")
        
        if hasattr(render_set, 'metadata'):
            metadata = render_set.metadata
            if 'complexity_score' in metadata:
                st.metric("Complexity Score", f"{metadata['complexity_score']}/10")

def display_coordinate_solution(coordinate_solution):
    """Display coordinate solution results from symbolic geometry planner."""
    
    st.markdown("## 🔢 Coordinate Solution Results")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Solved Objects", len(coordinate_solution.object_coordinates))
    with col2:
        st.metric("Constraint Solutions", len(coordinate_solution.constraint_solutions))
    with col3:
        accuracy = coordinate_solution.accuracy_metrics.get('confidence', 0.0)
        st.metric("Solution Confidence", f"{accuracy:.2f}")
    with col4:
        status = getattr(coordinate_solution, 'status', 'Unknown')
        st.metric("Status", status.value if hasattr(status, 'value') else str(status))
    
    # Coordinate System Info
    if hasattr(coordinate_solution, 'coordinate_system'):
        st.markdown("### 📍 Coordinate System")
        coord_sys = coordinate_solution.coordinate_system
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Origin:** {coord_sys.get('origin', 'Unknown')}")
        with col2:
            st.write(f"**Units:** {coord_sys.get('units', 'Unknown')}")
        with col3:
            st.write(f"**X-Axis:** {coord_sys.get('x_axis', 'Unknown')}")
    
    # Solved Object Coordinates
    if coordinate_solution.object_coordinates:
        st.markdown("### 🎯 Solved Object Coordinates")
        for obj_id, coords in coordinate_solution.object_coordinates.items():
            with st.expander(f"Object: {obj_id[:8]}..."):
                st.json(coords)
    
    # Constraint Verification
    if coordinate_solution.constraint_solutions:
        st.markdown("### ✅ Constraint Verification")
        for constraint_id, solution in coordinate_solution.constraint_solutions.items():
            satisfied = solution.get('satisfied', False)
            constraint_type = solution.get('type', 'Unknown')
            icon = "✅" if satisfied else "❌"
            
            with st.expander(f"{icon} {constraint_type.title()} - {constraint_id}"):
                st.write(f"**Satisfied:** {satisfied}")
                if 'mathematical_proof' in solution:
                    st.write(f"**Mathematical Proof:** {solution['mathematical_proof']}")
                if 'tolerance' in solution:
                    st.write(f"**Tolerance:** {solution['tolerance']}")
    
    # Mathematical Derivation
    if hasattr(coordinate_solution, 'mathematical_derivation') and coordinate_solution.mathematical_derivation:
        st.markdown("### 🔢 Mathematical Derivation")
        with st.expander("📝 Step-by-Step Mathematical Solution", expanded=False):
            st.markdown(coordinate_solution.mathematical_derivation)

    # Accuracy Metrics
    if hasattr(coordinate_solution, 'accuracy_metrics'):
        st.markdown("### 📊 Solution Quality")
        metrics = coordinate_solution.accuracy_metrics
        
        col1, col2, col3 = st.columns(3)
        with col1:
            precision = metrics.get('precision', 'Unknown')
            st.metric("Precision", precision)
        with col2:
            all_satisfied = metrics.get('all_constraints_satisfied', False)
            st.metric("All Constraints Satisfied", "✅" if all_satisfied else "❌")
        with col3:
            confidence = metrics.get('confidence', 0.0)
            st.metric("Mathematical Confidence", f"{confidence:.2f}")

def display_geometry_results(geometry_spec):
    """Display the initial geometry analysis results."""
    
    st.markdown("## 📐 Geometry Analysis Results")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Objects Detected", len(geometry_spec.objects))
    with col2:
        st.metric("Constraints Found", len(geometry_spec.constraints))
    with col3:
        confidence = getattr(geometry_spec, 'confidence', 1.0)
        st.metric("Overall Confidence", f"{confidence:.2f}")
    with col4:
        status = getattr(geometry_spec, 'status', Status.DRAFT)
        st.metric("Status", status.value if hasattr(status, 'value') else str(status))
    
    # Objects
    if geometry_spec.objects:
        st.markdown("### 🔷 Detected Objects")
        for i, obj in enumerate(geometry_spec.objects):
            with st.expander(f"Object {i+1}: {obj.type.title()} (ID: {obj.id[:8]})"):
                st.json(obj.properties)
    
    # Constraints  
    if geometry_spec.constraints:
        st.markdown("### 🔗 Detected Constraints")
        for i, constraint in enumerate(geometry_spec.constraints):
            with st.expander(f"Constraint {i+1}: {constraint.type.title()}"):
                st.write(f"**Objects:** {constraint.objects}")
                if hasattr(constraint, 'parameters') and constraint.parameters:
                    st.write(f"**Parameters:** {constraint.parameters}")

def main():
    """Main Streamlit application with O3 thinking visualization."""
    
    st.set_page_config(
        page_title="Gemini 2.5 Pro Geometry Pipeline", 
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Gemini 2.5 Pro Geometry Pipeline")
    st.markdown("*Experience step-by-step AI reasoning for geometric understanding with Google's Gemini*")
    
    # Sidebar for API key
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Check for API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            st.success("✅ Google API Key Detected")
        else:
            st.error("❌ Google API Key Not Found")
            st.markdown("Please set your `GOOGLE_API_KEY` environment variable")
    
    # Main input section
    st.markdown("### 💬 Prompt Input")
    user_prompt = st.text_area(
        "Describe the geometric shape or relationship you want to create:",
        placeholder="Example: Draw a square that fits inside a circle of 10cm. The vertices of square should lie on the circle.",
        height=100
    )
    
    # Image upload (optional)
    st.markdown("### 🖼️ Reference Image (Optional)")
    uploaded_file = st.file_uploader(
        "Upload a reference image for additional geometric analysis:",
        type=['png', 'jpg', 'jpeg'],
        help="The AI will analyze this image along with your text prompt"
    )
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Analyze with Gemini 2.5 Pro",
            type="primary",
            use_container_width=True
        )
    
    # Process the input
    if process_button and user_prompt.strip():
        with st.spinner("🧠 Gemini AI agents are thinking..."):
            try:
                # Create prompt bundle
                import uuid
                prompt_bundle = PromptBundle(
                    prompt_id=str(uuid.uuid4()),
                    text=user_prompt.strip()
                )
                
                # Handle image upload
                image_uri = None
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        image_uri = tmp_file.name
                    
                    # Add image to prompt bundle
                    prompt_bundle.images = [image_uri]
                    st.info(f"📸 Image uploaded: {uploaded_file.name} - Will be processed by Image Preprocessor and Vision Interpreter")
                
                # Process with orchestrator - use full pipeline
                from agents.orchestrator import _orchestrator
                result = _orchestrator.process_full_pipeline(prompt_bundle)
                
                # Clean up temp file
                if image_uri and os.path.exists(image_uri):
                    os.unlink(image_uri)
                
                # Display results
                if hasattr(result, 'pipeline_metadata') and result.pipeline_metadata.get('pipeline_status') == 'COMPLETE':
                    # Complete pipeline success - FinalAssets returned
                    st.success("🎉 Complete Gemini AI Pipeline Success!")
                    st.info("📊 All 6 agents completed successfully: Prompt → Vision → Image Preprocessor → Symbolic Planner → Layout Designer → Renderer → Math Verifier")
                    
                    # Get pipeline components
                    pipeline_data = result.pipeline_metadata
                    coordinate_solution = pipeline_data.get('coordinate_solution')
                    layout_plan = pipeline_data.get('layout_plan')
                    render_set = pipeline_data.get('render_set')
                    qa_report = pipeline_data.get('qa_report')
                    
                    # Display complete reasoning from all agents
                    if coordinate_solution and hasattr(coordinate_solution, 'agent_reasoning'):
                        display_pipeline_flow(coordinate_solution)
                    
                    # Display final SVG output
                    st.markdown("## 🎨 Final Rendered Output")
                    if result.final_svg:
                        st.markdown("### 📐 Generated Geometry")
                        st.components.v1.html(result.final_svg, height=400)
                        
                        with st.expander("📄 View SVG Source"):
                            st.code(result.final_svg, language='xml')
                    
                    # Display coordinate solution results
                    if coordinate_solution:
                        display_coordinate_solution(coordinate_solution)
                    
                    # Display verification results
                    if qa_report:
                        display_qa_report(qa_report)
                    
                    # Display layout and rendering details
                    if layout_plan and render_set:
                        display_layout_and_rendering_results(layout_plan, render_set)
                        
                elif hasattr(result, 'details') and 'coordinate_solution' in result.details:
                    # Partial pipeline (up to coordinate solution)
                    coordinate_solution = result.details['coordinate_solution']
                    
                    st.success("✅ Gemini AI Partial Pipeline Complete!")
                    st.info(f"📊 Pipeline Status: {result.message}")
                    
                    # Display the complete thinking process
                    display_pipeline_flow(coordinate_solution)
                    
                    # Display coordinate solution results
                    display_coordinate_solution(coordinate_solution)
                    
                elif hasattr(result, 'objects'):  # Geometry spec only
                    st.success("✅ Gemini AI Analysis Complete!")
                    
                    # Display the thinking process
                    display_pipeline_flow(result)
                    
                    # Display geometry results
                    display_geometry_results(result)
                    
                else:  # Error
                    st.error("❌ Analysis Failed")
                    st.write(f"Error: {result.message}")
                    if hasattr(result, 'details'):
                        st.json(result.details)
                        
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                logger.error(f"Frontend error: {e}")
    
    elif process_button:
        st.warning("⚠️ Please enter a prompt to analyze")

if __name__ == "__main__":
    main() 