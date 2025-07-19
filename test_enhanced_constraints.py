#!/usr/bin/env python3
"""
Test script for enhanced constraint detection.
"""

from agents.data_structures import PromptBundle
from agents import orchestrator
import uuid


def test_inscription_constraint():
    """Test the enhanced constraint detection for inscription."""
    
    print("🧪 Testing Enhanced Constraint Detection")
    print("=" * 60)
    
    test_prompts = [
        "Draw a square that fits inside a circle of 10cm. The vertices of square should lie on the circle.",
        "Create a triangle inscribed in a circle with radius 5cm",
        "Make a circle that surrounds a square",
        "Draw two lines that intersect at right angles",
        "Create a pentagon with vertices touching a circle"
    ]
    
    for i, prompt_text in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt_text}")
        print("-" * 50)
        
        try:
            prompt_bundle = PromptBundle(
                prompt_id=str(uuid.uuid4()),
                text=prompt_text,
                images=[],
                meta={"source": "constraint_test"}
            )
            
            result = orchestrator.handle(prompt_bundle)
            
            print(f"✅ Objects: {len(result.objects)}")
            print(f"✅ Constraints: {len(result.constraints)}")
            
            # Show detected shapes
            for j, obj in enumerate(result.objects, 1):
                print(f"   Shape {j}: {obj.type} (ID: {obj.id})")
            
            # Show detected constraints
            for j, constraint in enumerate(result.constraints, 1):
                print(f"   Constraint {j}: {constraint.type}")
                print(f"      Objects: {constraint.objects}")
                if constraint.parameters:
                    print(f"      Parameters: {constraint.parameters}")
            
            if not result.constraints:
                print("   ⚠️  No constraints detected")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 Enhanced Constraint Testing Complete!")


if __name__ == "__main__":
    test_inscription_constraint() 