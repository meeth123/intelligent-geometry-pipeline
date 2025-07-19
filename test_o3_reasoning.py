#!/usr/bin/env python3
"""
Test script to showcase Gemini 2.5 Pro reasoning process from each agent.
"""
import os
from agents.data_structures import PromptBundle
from agents.orchestrator import handle

def display_gemini_reasoning(geometry_spec):
    """Display the Gemini reasoning from all agents."""
    
    print("\n" + "="*80)
    print("🧠 GEMINI AI THINKING PROCESS ANALYSIS")
    print("="*80)
    
    # Show agent reasoning
    if hasattr(geometry_spec, 'agent_reasoning') and geometry_spec.agent_reasoning:
        print("\n💭 AGENT REASONING:")
        print("-" * 50)
        
        for agent, reasoning in geometry_spec.agent_reasoning.items():
            print(f"\n🤖 {agent.upper().replace('_', ' ')}:")
            print("─" * 40)
            print(reasoning)
    
    # Show processing steps
    if hasattr(geometry_spec, 'processing_steps') and geometry_spec.processing_steps:
        print("\n📊 PROCESSING STEPS:")
        print("-" * 50)
        
        for i, step in enumerate(geometry_spec.processing_steps, 1):
            agent = step.get('agent', 'unknown')
            step_name = step.get('step', 'unknown')
            reasoning = step.get('reasoning', 'No reasoning provided')
            model = step.get('model', 'unknown')
            
            print(f"\n{i}. {agent.upper().replace('_', ' ')} - {step_name}")
            print(f"   Model: {model}")
            
            # Show metrics if available
            metrics = []
            if 'objects_found' in step:
                metrics.append(f"Objects: {step['objects_found']}")
            if 'constraints_found' in step:
                metrics.append(f"Constraints: {step['constraints_found']}")
            if 'confidence' in step:
                metrics.append(f"Confidence: {step['confidence']:.2f}")
            
            if metrics:
                print(f"   Metrics: {', '.join(metrics)}")
            
            print("   Reasoning:")
            print("   " + reasoning.replace('\n', '\n   '))

def test_complex_prompts():
    """Test Gemini reasoning on various complex geometric prompts."""
    
    test_prompts = [
        "Draw a square that fits inside a circle of 10cm. The vertices of square should lie on the circle.",
        "Create two parallel lines that are 5cm apart, with a perpendicular line connecting them.",
        "Draw an equilateral triangle inscribed in a circle of radius 8cm.",
        "Create a rectangle with width twice its height, positioned tangent to a circle."
    ]
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("⚠️  Google API key not found. Using fallback analysis.")
        print("   Set GOOGLE_API_KEY environment variable for full Gemini reasoning.")
        print("")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*20} TEST {i} {'='*20}")
        print(f"📝 PROMPT: {prompt}")
        
        # Create prompt bundle
        import uuid
        prompt_bundle = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=prompt
        )
        
        # Process with orchestrator
        result = handle(prompt_bundle)
        
        if hasattr(result, 'objects'):
            print(f"\n✅ SUCCESS: {len(result.objects)} objects, {len(result.constraints)} constraints")
            
            # Display Gemini reasoning
            display_gemini_reasoning(result)
            
        else:
            print(f"\n❌ ERROR: {result.message}")
        
        print("\n" + "─"*80)

if __name__ == "__main__":
    print("🧠 GEMINI AI REASONING TEST")
    print("Testing step-by-step thinking process of AI agents")
    test_complex_prompts() 