#!/usr/bin/env python3
"""
Test the Complete 6-Agent Gemini 2.5 Pro Pipeline
Layout Designer + Renderer + Math Consistency Verifier
"""

import os
import sys
import uuid
from agents.data_structures import PromptBundle, AgentError
from agents.orchestrator import _orchestrator

def test_complete_six_agent_pipeline():
    """Test the complete pipeline with all 6 agents"""
    
    print("🚀 COMPLETE 6-AGENT GEMINI 2.5 PRO PIPELINE TEST")
    print("Testing: Prompt Interpreter → Image Preprocessor → Vision Interpreter → Symbolic Geometry Planner → Layout Designer → Renderer → Math Consistency Verifier")
    print("=" * 100)
    
    # Test prompts optimized for the complete pipeline
    test_prompts = [
        "Draw a square inscribed in a circle of radius 10cm with precise coordinates"
    
    ]
    
    for i, prompt_text in enumerate(test_prompts, 1):
        print(f"\n{'='*30} TEST {i} {'='*30}")
        print(f"📝 PROMPT: {prompt_text}")
        
        try:
            # Create prompt bundle
            prompt_bundle = PromptBundle(
                prompt_id=str(uuid.uuid4()),
                text=prompt_text
            )
            
            print(f"🎯 Running complete 6-agent pipeline...")
            
            # Process with complete pipeline
            result = _orchestrator.process_full_pipeline(prompt_bundle)
            
            if isinstance(result, AgentError):
                print(f"❌ PIPELINE ERROR: {result.message}")
                if hasattr(result, 'details'):
                    print(f"   Details: {result.details}")
                continue
            
            # Check if we got FinalAssets (complete success)
            if hasattr(result, 'pipeline_metadata'):
                pipeline_data = result.pipeline_metadata
                pipeline_status = pipeline_data.get('pipeline_status', 'Unknown')
                
                if pipeline_status == 'COMPLETE':
                    print(f"🎉 COMPLETE PIPELINE SUCCESS!")
                    
                    # Display comprehensive results
                    print(f"\n📊 COMPLETE PIPELINE RESULTS:")
                    print(f"   🎯 Total Agents: {pipeline_data.get('total_agents', 0)}")
                    
                    # Agent reasoning summary
                    agent_reasoning = pipeline_data.get('agent_reasoning', {})
                    print(f"\n🧠 AGENT REASONING CAPTURED:")
                    for agent, reasoning in agent_reasoning.items():
                        icon = {
                            'prompt_interpreter': '🧠',
                            'image_preprocessor': '🖼️',
                            'vision_interpreter': '👁️',
                            'symbolic_geometry_planner': '🔢',
                            'layout_designer': '🎨',
                            'renderer': '🖨️',
                            'math_consistency_verifier': '✅',
                            'orchestrator': '🎯'
                        }.get(agent, '🤖')
                        
                        print(f"   {icon} {agent.upper()}: {len(reasoning)} characters")
                    
                    # Coordinate Solution Results
                    coordinate_solution = pipeline_data.get('coordinate_solution')
                    if coordinate_solution:
                        print(f"\n🔢 COORDINATE SOLUTION:")
                        print(f"   📍 Solved Objects: {len(coordinate_solution.object_coordinates)}")
                        print(f"   📏 Constraint Solutions: {len(coordinate_solution.constraint_solutions)}")
                        if hasattr(coordinate_solution, 'accuracy_metrics'):
                            confidence = coordinate_solution.accuracy_metrics.get('confidence', 0)
                            print(f"   💯 Solution Confidence: {confidence}")
                    
                    # Layout Plan Results
                    layout_plan = pipeline_data.get('layout_plan')
                    if layout_plan:
                        print(f"\n🎨 LAYOUT DESIGN:")
                        print(f"   📐 SVG Size: {len(layout_plan.svg)} characters")
                        print(f"   🎨 Style Tokens: {len(layout_plan.style_tokens)}")
                        print(f"   🏷️ Labels: {len(layout_plan.labels)}")
                    
                    # Render Set Results
                    render_set = pipeline_data.get('render_set')
                    if render_set:
                        print(f"\n🖨️ RENDERING:")
                        print(f"   📄 Final SVG: {len(render_set.render_svg)} characters")
                        if hasattr(render_set, 'rendering_decisions'):
                            optimization = render_set.rendering_decisions.get('optimization_type', 'unknown')
                            print(f"   ⚙️ Optimization: {optimization}")
                    
                    # QA Report Results
                    qa_report = pipeline_data.get('qa_report')
                    if qa_report:
                        print(f"\n✅ MATH VERIFICATION:")
                        print(f"   📊 Status: {qa_report.status}")
                        print(f"   🎯 Tolerance: {qa_report.tolerance_mm}mm")
                        print(f"   ⚠️ Issues Found: {len(qa_report.issues)}")
                        if hasattr(qa_report, 'verification_data'):
                            quality_score = qa_report.verification_data.get('overall_quality_score', 0)
                            print(f"   💯 Quality Score: {quality_score:.2f}/1.0")
                    
                    # Final Assets
                    print(f"\n🎁 FINAL ASSETS:")
                    print(f"   📐 Final SVG: {'✅ Generated' if result.final_svg else '❌ Missing'}")
                    print(f"   🖼️ Final PNG: {'✅ Generated' if result.final_png else '⚠️ Not generated'}")
                    
                else:
                    print(f"⚠️ PIPELINE STATUS: {pipeline_status}")
                    
            else:
                print(f"❌ UNEXPECTED RESULT TYPE: {type(result)}")
                print(f"   Result has attributes: {list(vars(result).keys()) if hasattr(result, '__dict__') else 'No attributes'}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
        
        print("─" * 100)
    
    print(f"\n🎉 COMPLETE 6-AGENT PIPELINE TEST FINISHED!")
    print(f"💡 TIP: Check the frontend for detailed visual results!")
    print(f"🔧 Run: streamlit run frontend.py")

if __name__ == "__main__":
    test_complete_six_agent_pipeline() 