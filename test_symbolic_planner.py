#!/usr/bin/env python3
"""
Test script for the Symbolic Geometry Planner Agent.
Tests mathematical constraint solving with Gemini 2.5 Pro.
"""
import os
import uuid
from agents.data_structures import PromptBundle, GeometrySpec
from agents.orchestrator import handle
from agents import symbolic_geometry_planner

def test_symbolic_geometry_planner():
    """Test the symbolic geometry planner with various geometric scenarios."""
    
    print("🧠 SYMBOLIC GEOMETRY PLANNER TEST")
    print("Testing mathematical constraint solving with Gemini 2.5 Pro")
    print("="*80)
    
    # Test cases
    test_prompts = [
        "Draw a square that fits inside a circle of 10cm. The vertices of square should lie on the circle.",
        "Create two parallel lines that are 5cm apart, with a perpendicular line connecting them.",
        "Draw an equilateral triangle inscribed in a circle of radius 8cm."
    ]
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("⚠️  Google API key not found. Testing with fallback analysis.")
        print("   Set GOOGLE_API_KEY environment variable for full Gemini reasoning.")
        print("")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*20} TEST {i} {'='*20}")
        print(f"📝 PROMPT: {prompt}")
        
        try:
            # Create prompt bundle and get geometry spec
            prompt_bundle = PromptBundle(
                prompt_id=str(uuid.uuid4()),
                text=prompt
            )
            
            # Get geometry spec from orchestrator
            geometry_spec = handle(prompt_bundle)
            
            if hasattr(geometry_spec, 'objects'):
                print(f"\n📐 GEOMETRY SPEC:")
                print(f"   Objects: {len(geometry_spec.objects)}")
                print(f"   Constraints: {len(geometry_spec.constraints)}")
                
                # Test the symbolic geometry planner directly
                print(f"\n🔢 TESTING SYMBOLIC GEOMETRY PLANNER:")
                coordinate_solution = symbolic_geometry_planner.handle(geometry_spec)
                
                if hasattr(coordinate_solution, 'object_coordinates'):
                    print(f"✅ SUCCESS!")
                    print(f"   Solved Objects: {len(coordinate_solution.object_coordinates)}")
                    print(f"   Constraint Solutions: {len(coordinate_solution.constraint_solutions)}")
                    print(f"   Status: {coordinate_solution.status}")
                    
                    # Show mathematical reasoning
                    if hasattr(coordinate_solution, 'mathematical_derivation'):
                        reasoning = coordinate_solution.mathematical_derivation
                        print(f"\n🧠 MATHEMATICAL REASONING:")
                        print(f"   Length: {len(reasoning)} characters")
                        print(f"   Preview: {reasoning[:200]}...")
                    
                    # Show coordinate details
                    print(f"\n📍 COORDINATE DETAILS:")
                    for obj_id, coords in coordinate_solution.object_coordinates.items():
                        print(f"   {obj_id}: {coords}")
                    
                    # Show constraint verification
                    if coordinate_solution.constraint_solutions:
                        print(f"\n✅ CONSTRAINT VERIFICATION:")
                        for constraint_id, solution in coordinate_solution.constraint_solutions.items():
                            satisfied = solution.get('satisfied', False)
                            print(f"   {constraint_id}: {'✅' if satisfied else '❌'} {solution.get('type', 'unknown')}")
                    
                    # Show accuracy metrics
                    if hasattr(coordinate_solution, 'accuracy_metrics'):
                        metrics = coordinate_solution.accuracy_metrics
                        print(f"\n📊 ACCURACY METRICS:")
                        print(f"   Precision: {metrics.get('precision', 'unknown')}")
                        print(f"   All Constraints Satisfied: {metrics.get('all_constraints_satisfied', False)}")
                        print(f"   Confidence: {metrics.get('confidence', 0.0):.2f}")
                
                else:
                    print(f"❌ ERROR: {coordinate_solution.message}")
                
            else:
                print(f"❌ ERROR: Could not get geometry spec - {geometry_spec.message}")
        
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
        
        print("\n" + "─"*80)

if __name__ == "__main__":
    test_symbolic_geometry_planner() 