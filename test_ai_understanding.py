#!/usr/bin/env python3
"""
Test the AI-based prompt interpreter's understanding of geometric relationships.
"""

from agents.data_structures import PromptBundle
from agents import orchestrator
import uuid
import os

def test_problematic_prompt():
    """Test the specific prompt that was failing before."""
    
    print("🧪 Testing AI-Based Geometric Understanding")
    print("=" * 60)
    
    # Set API key if not already set
    if not os.getenv('OPENAI_API_KEY'):
        api_key = "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"
        os.environ['OPENAI_API_KEY'] = api_key
    
    # The exact prompt that was problematic
    prompt_text = "Draw a square that fits inside a circle of 10cm. The vertices of square should lie on the circle."
    
    print(f"📝 Testing prompt: {prompt_text}")
    print("-" * 60)
    
    try:
        prompt_bundle = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=prompt_text,
            images=[],
            meta={"source": "ai_test"}
        )
        
        result = orchestrator.handle(prompt_bundle)
        
        print(f"✅ SUCCESS!")
        print(f"   Objects: {len(result.objects)}")
        print(f"   Constraints: {len(result.constraints)}")
        print()
        
        # Show objects
        print("📦 Objects detected:")
        for i, obj in enumerate(result.objects, 1):
            print(f"   {i}. {obj.type.title()} (ID: {obj.id})")
            for key, value in obj.properties.items():
                print(f"      {key}: {value}")
        print()
        
        # Show constraints (this was the problem!)
        print("🔗 Constraints detected:")
        if result.constraints:
            for i, constraint in enumerate(result.constraints, 1):
                print(f"   {i}. {constraint.type}")
                print(f"      Objects: {constraint.objects}")
                if constraint.parameters:
                    print(f"      Parameters: {constraint.parameters}")
        else:
            print("   ❌ NO CONSTRAINTS DETECTED (this was the problem)")
        print()
        
        # Show AI analysis
        print("🤖 AI Analysis:")
        annotations = result.annotations
        print(f"   AI Model: {annotations.get('ai_model', 'N/A')}")
        print(f"   AI Analysis: {annotations.get('ai_analysis', False)}")
        if annotations.get('extracted_dimensions'):
            print(f"   Dimensions: {annotations['extracted_dimensions']}")
        
        # Expected result analysis
        print("\n🎯 Expected vs Actual:")
        print("   Expected Objects: Circle + Square")
        print("   Expected Constraints: 'inscribed' or 'tangent'")
        print("   Expected Circle: radius ~5cm (from 10cm diameter)")
        
        success = len(result.constraints) > 0
        print(f"\n{'✅ PASS' if success else '❌ FAIL'}: {'Constraints detected!' if success else 'Still no constraints detected'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_problematic_prompt() 