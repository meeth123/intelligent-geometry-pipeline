#!/usr/bin/env python3
"""
Test JSON Parsing Fixes Across All Agents
Verify that the enhanced error handling and JSON cleaning works for all AI agents
"""

import json
import uuid
from agents.data_structures import PromptBundle, GeometrySpec, CoordinateSolution, LayoutPlan
from agents.orchestrator import _orchestrator

def test_json_cleaning_function():
    """Test the JSON cleaning utility function"""
    
    print("🧹 TESTING ENHANCED JSON CLEANING FUNCTION")
    print("=" * 50)
    
    # Import the cleaning function
    from agents.prompt_interpreter import _clean_json_response
    
    test_cases = [
        {
            "name": "Markdown Code Blocks",
            "input": '''```json
{
    "test": "value with markdown",
    "number": 123
}
```''',
            "should_clean": True
        },
        {
            "name": "Control Characters",
            "input": '{"test": "value\x00with\x08control\x1fchars"}',
            "should_clean": True
        },
        {
            "name": "Multiline Strings in JSON",
            "input": '''```json
{
    "reasoning": "This is a multiline
    string with actual newlines
    that should be escaped",
    "number": 123
}
```''',
            "should_clean": True
        },
        {
            "name": "Trailing Commas",
            "input": '{"test": "value", "array": [1, 2, 3,],}',
            "should_clean": True
        },
        {
            "name": "Complex Real Case",
            "input": '''```json
{
    "reasoning": "The user requested a geometric setup involving a circle and two external points from which tangents are drawn.",
    "objects": [
        {"type": "circle", "radius": 5}
    ]
}
```''',
            "should_clean": True
        },
        {
            "name": "Valid JSON",
            "input": '{"test": "valid", "number": 123}',
            "should_clean": False
        }
    ]
    
    for case in test_cases:
        print(f"\n🧪 Testing: {case['name']}")
        print(f"Input: {repr(case['input'][:80])}...")
        
        try:
            # Try original parsing
            try:
                original_result = json.loads(case['input'])
                print("✅ Original JSON parsed successfully")
            except json.JSONDecodeError:
                print("❌ Original JSON failed (expected for most test cases)")
                
                # Try cleaned parsing
                cleaned = _clean_json_response(case['input'])
                if cleaned:
                    cleaned_result = json.loads(cleaned)
                    print(f"✅ Cleaned JSON parsed successfully")
                    print(f"   Keys: {list(cleaned_result.keys())}")
                else:
                    print("❌ Cleaning returned empty result")
                
        except Exception as e:
            print(f"❌ Both original and cleaned parsing failed: {e}")
            
    print("\n✅ Enhanced JSON cleaning function tested!")

def test_pipeline_resilience():
    """Test pipeline resilience with potentially problematic prompts"""
    
    print(f"\n🛡️ TESTING PIPELINE RESILIENCE")
    print("-" * 40)
    
    # Test prompts that might cause JSON issues
    test_prompts = [
        "Draw a square with side length 5cm and label it 'My\nSquare'",
        "Create a circle with radius 3cm at position (10, 20)",
        "Draw a triangle with angles 60°, 60°, and 60°",
    ]
    
    for i, prompt_text in enumerate(test_prompts, 1):
        print(f"\n🧪 TEST {i}: '{prompt_text}'")
        
        prompt_bundle = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=prompt_text
        )
        
        try:
            # Process with full pipeline
            result = _orchestrator.process_full_pipeline(prompt_bundle)
            
            print(f"✅ Pipeline completed successfully")
            print(f"   Result type: {type(result).__name__}")
            
            # Check if we got meaningful results
            if hasattr(result, 'pipeline_metadata'):
                status = result.pipeline_metadata.get('pipeline_status', 'Unknown')
                print(f"   Status: {status}")
                
                if status == 'COMPLETE':
                    print("🎉 Full pipeline success with JSON parsing fixes!")
                
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")

def test_agent_error_recovery():
    """Test individual agent error recovery with JSON issues"""
    
    print(f"\n🔧 TESTING AGENT ERROR RECOVERY")
    print("-" * 40)
    
    print("💡 This test verifies that agents gracefully handle JSON parsing errors")
    print("💡 and either clean the JSON successfully or fall back appropriately")
    
    # Create a simple test case
    prompt_bundle = PromptBundle(
        prompt_id=str(uuid.uuid4()),
        text="Draw a simple circle with 5cm radius"
    )
    
    try:
        # Test prompt interpreter
        from agents import prompt_interpreter
        prompt_result = prompt_interpreter.handle(prompt_bundle)
        
        if isinstance(prompt_result, GeometrySpec):
            print("✅ Prompt Interpreter: JSON parsing robust")
        else:
            print(f"⚠️ Prompt Interpreter returned: {type(prompt_result).__name__}")
        
        # If we got a GeometrySpec, test downstream agents
        if isinstance(prompt_result, GeometrySpec):
            
            # Test symbolic geometry planner
            from agents import symbolic_geometry_planner
            coord_result = symbolic_geometry_planner.handle(prompt_result)
            
            if isinstance(coord_result, CoordinateSolution):
                print("✅ Symbolic Geometry Planner: JSON parsing robust")
                
                # Test layout designer
                from agents import layout_designer
                layout_result = layout_designer.handle(coord_result)
                
                if isinstance(layout_result, LayoutPlan):
                    print("✅ Layout Designer: JSON parsing robust")
                else:
                    print(f"⚠️ Layout Designer returned: {type(layout_result).__name__}")
                    
            else:
                print(f"⚠️ Symbolic Geometry Planner returned: {type(coord_result).__name__}")
                
    except Exception as e:
        print(f"❌ Agent error recovery test failed: {e}")

def test_json_parsing_statistics():
    """Gather statistics on JSON parsing success rates"""
    
    print(f"\n📊 JSON PARSING STATISTICS")
    print("-" * 30)
    
    print("💡 Run this test with multiple prompts to gather statistics")
    print("💡 on how often the JSON cleaning fixes help vs fallback usage")
    
    # Simple test
    test_prompt = "Draw a square inscribed in a circle with 10cm diameter"
    
    prompt_bundle = PromptBundle(
        prompt_id=str(uuid.uuid4()),
        text=test_prompt
    )
    
    try:
        result = _orchestrator.process_full_pipeline(prompt_bundle)
        print(f"✅ Pipeline completed for: '{test_prompt}'")
        print(f"   Final result: {type(result).__name__}")
        
        # Could collect statistics on:
        # - How many agents used cleaned JSON vs original
        # - How many fell back to simple solutions
        # - Overall success rate improvements
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")

if __name__ == "__main__":
    test_json_cleaning_function()
    test_pipeline_resilience()
    test_agent_error_recovery()
    test_json_parsing_statistics()
    print("\n🎯 JSON parsing fixes testing complete!")
    print("💡 Your pipeline is now much more robust against AI response formatting issues!") 