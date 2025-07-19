#!/usr/bin/env python3
"""
Test the complete Gemini 2.5 Pro pipeline with all agents including
Image Preprocessor, Vision Interpreter, and Symbolic Geometry Planner
"""

import os
import sys
import uuid
from agents.data_structures import PromptBundle, AgentError
from agents.orchestrator import _orchestrator

def test_complete_pipeline():
    """Test the complete pipeline with all agents"""
    
    print("🔥 COMPLETE GEMINI 2.5 PRO PIPELINE TEST")
    print("Testing all agents: Prompt Interpreter → Vision Interpreter → Image Preprocessor → Symbolic Geometry Planner")
    print("=" * 80)
    
    # Test prompts that should work well with symbolic geometry
    test_prompts = [
        "Draw a square that fits inside a circle of radius 10cm. The square vertices should touch the circle.",
        "Create two parallel lines that are 5cm apart with a perpendicular line connecting them.",
        "Draw an equilateral triangle inscribed in a circle of radius 8cm."
    ]
    
    for i, prompt_text in enumerate(test_prompts, 1):
        print(f"\n{'='*20} TEST {i} {'='*20}")
        print(f"📝 PROMPT: {prompt_text}")
        
        try:
            # Create prompt bundle
            prompt_bundle = PromptBundle(
                prompt_id=str(uuid.uuid4()),
                text=prompt_text
            )
            
            print(f"🎯 Processing with orchestrator...")
            
            # Process with full pipeline
            result = _orchestrator.process_full_pipeline(prompt_bundle)
            
            if isinstance(result, AgentError):
                print(f"❌ ERROR: {result.message}")
                continue
            
            # Check if we got coordinate solution
            if hasattr(result, 'details') and 'coordinate_solution' in result.details:
                coordinate_solution = result.details['coordinate_solution']
                print(f"✅ FULL PIPELINE SUCCESS!")
                
                # Display agent reasoning
                if hasattr(coordinate_solution, 'agent_reasoning'):
                    print(f"\n🧠 AGENT REASONING CAPTURED:")
                    for agent, reasoning in coordinate_solution.agent_reasoning.items():
                        icon = {
                            'prompt_interpreter': '🧠',
                            'image_preprocessor': '🖼️',
                            'vision_interpreter': '👁️',
                            'symbolic_geometry_planner': '🔢',
                            'orchestrator': '🎯'
                        }.get(agent, '🤖')
                        
                        print(f"   {icon} {agent.upper()}: {len(reasoning)} characters of reasoning")
                
                # Display results
                print(f"\n📊 SOLUTION RESULTS:")
                print(f"   🎯 Solved Objects: {len(coordinate_solution.object_coordinates)}")
                print(f"   📏 Constraint Solutions: {len(coordinate_solution.constraint_solutions)}")
                
                if hasattr(coordinate_solution, 'accuracy_metrics'):
                    confidence = coordinate_solution.accuracy_metrics.get('confidence', 0)
                    print(f"   💯 Confidence: {confidence}")
                
                print(f"   ✅ Status: {coordinate_solution.status}")
                
            elif hasattr(result, 'objects'):
                # Partial success - geometry spec only
                print(f"✅ PARTIAL SUCCESS (Geometry Spec)")
                print(f"   🔷 Objects: {len(result.objects)}")
                print(f"   🔗 Constraints: {len(result.constraints)}")
                
                if hasattr(result, 'agent_reasoning'):
                    print(f"   🧠 Agent reasoning captured: {len(result.agent_reasoning)} agents")
            else:
                print(f"❌ UNEXPECTED RESULT: {type(result)}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
        
        print("─" * 80)
    
    print(f"\n🎉 PIPELINE TEST COMPLETE!")
    print(f"💡 TIP: Start the frontend with 'streamlit run frontend.py' to see detailed reasoning!")

if __name__ == "__main__":
    test_complete_pipeline() 