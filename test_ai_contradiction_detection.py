#!/usr/bin/env python3
"""
Test AI-Powered Contradiction Detection in Prompt Interpreter
"""

import uuid
from agents.data_structures import PromptBundle, ClarificationRequest
from agents.prompt_interpreter import handle

def test_ai_contradiction_detection():
    """Test AI-powered logical inconsistency detection"""
    
    print("🧠 TESTING AI-POWERED CONTRADICTION DETECTION")
    print("=" * 60)
    
    # Test cases with various types of logical issues
    test_cases = [
        {
            "name": "Mathematical Contradiction - Impossible Inscription",
            "prompt": "Draw a square with 15cm sides inscribed in a circle with 10cm diameter",
            "should_detect": True,
            "expected_category": "MATHEMATICAL_CONTRADICTION"
        },
        {
            "name": "Logical Inconsistency - Inside vs Outside",
            "prompt": "Create a circle inside a square but also outside the square at the same time",
            "should_detect": True,
            "expected_category": "LOGICAL_INCONSISTENCY"
        },
        {
            "name": "Ambiguous Reference - Multiple Circles",
            "prompt": "Draw two circles, then move the circle to the right",
            "should_detect": True,
            "expected_category": "AMBIGUOUS_REFERENCE"
        },
        {
            "name": "Missing Information - No Dimensions",
            "prompt": "Create a square and a circle",
            "should_detect": True,
            "expected_category": "MISSING_INFORMATION"
        },
        {
            "name": "Impossible Construction - Negative Size",
            "prompt": "Draw a circle with -5cm radius",
            "should_detect": True,
            "expected_category": "IMPOSSIBLE_CONSTRUCTION"
        },
        {
            "name": "Terminology Inconsistency - 2D/3D Mix",
            "prompt": "Draw a circle and place a cube on top of it",
            "should_detect": True,
            "expected_category": "TERMINOLOGY_INCONSISTENCY"
        },
        {
            "name": "Clear Prompt - No Issues",
            "prompt": "Draw a square with 5cm sides centered at coordinates (0,0)",
            "should_detect": False,
            "expected_category": "CLEAR"
        },
        {
            "name": "Valid Inscribed Geometry",
            "prompt": "Draw a square inscribed in a circle with 14cm diameter",
            "should_detect": False,
            "expected_category": "CLEAR"
        },
        {
            "name": "Complex But Clear Prompt",
            "prompt": "Create two parallel lines 3cm apart, then draw a circle with 2cm radius tangent to both lines",
            "should_detect": False,
            "expected_category": "CLEAR"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test_case['name']}")
        print(f"Prompt: '{test_case['prompt']}'")
        print("-" * 50)
        
        # Create prompt bundle
        prompt_bundle = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=test_case['prompt']
        )
        
        try:
            # Process prompt with AI analysis
            result = handle(prompt_bundle)
            
            # Check result type
            is_clarification = isinstance(result, ClarificationRequest)
            
            if test_case['should_detect']:
                if is_clarification:
                    print("✅ PASS: AI correctly detected issues")
                    
                    # Show AI analysis details
                    print(f"   📋 Category: {result.contradiction_type}")
                    print(f"   📋 Issues: {len(result.detected_issues)}")
                    for issue in result.detected_issues:
                        print(f"      • {issue}")
                    
                    print(f"   ❓ Questions: {len(result.clarification_questions)}")
                    for question in result.clarification_questions:
                        print(f"      • {question}")
                    
                    if result.suggested_resolutions:
                        print(f"   💡 Suggestions: {len(result.suggested_resolutions)}")
                        for suggestion in result.suggested_resolutions:
                            print(f"      • {suggestion}")
                    
                    print(f"   🧠 AI Reasoning:")
                    reasoning = result.agent_reasoning
                    print(f"      {reasoning[:200]}..." if len(reasoning) > 200 else f"      {reasoning}")
                    
                    # Check if category matches expectation
                    if result.contradiction_type == test_case['expected_category']:
                        print(f"   ✅ Correct category: {result.contradiction_type}")
                    else:
                        print(f"   ⚠️ Expected {test_case['expected_category']}, got {result.contradiction_type}")
                    
                    passed_tests += 1
                else:
                    print("❌ FAIL: AI should have detected issues but didn't")
                    print(f"   Got result type: {type(result).__name__}")
            else:
                if not is_clarification:
                    print("✅ PASS: AI correctly found no issues")
                    passed_tests += 1
                else:
                    print("❌ FAIL: AI false positive - detected issues where none exist")
                    print(f"   Detected category: {result.contradiction_type}")
                    print(f"   Issues: {result.detected_issues}")
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All AI contradiction detection tests passed!")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Most tests passed - AI analysis working well!")
    else:
        print("⚠️ Several tests failed - check AI prompt or implementation")

def test_ai_reasoning_quality():
    """Test the quality of AI reasoning for different prompt types"""
    
    print(f"\n🎯 TESTING AI REASONING QUALITY")
    print("-" * 40)
    
    complex_prompt = """
    Draw a triangle with angles 90°, 45°, and 50°. Then inscribe a circle inside it.
    The triangle should have one side of 10cm, and the inscribed circle should have a radius of 3cm.
    Place another circle outside the triangle that touches all three sides.
    """
    
    prompt_bundle = PromptBundle(
        prompt_id=str(uuid.uuid4()),
        text=complex_prompt
    )
    
    print(f"Testing complex prompt:")
    print(f"'{complex_prompt}'")
    
    result = handle(prompt_bundle)
    
    if isinstance(result, ClarificationRequest):
        print("✅ AI detected issues in complex prompt")
        
        print(f"\n🎯 AI ANALYSIS QUALITY:")
        print(f"Category: {result.contradiction_type}")
        print(f"Issues Found: {len(result.detected_issues)}")
        
        print(f"\n📋 Detailed Issues:")
        for i, issue in enumerate(result.detected_issues, 1):
            print(f"{i}. {issue}")
        
        print(f"\n❓ Clarification Questions:")
        for i, question in enumerate(result.clarification_questions, 1):
            print(f"{i}. {question}")
        
        print(f"\n🧠 AI Reasoning:")
        print(result.agent_reasoning)
        
        # Check reasoning quality indicators
        reasoning_quality = []
        reasoning_text = result.agent_reasoning.lower()
        
        if 'angle' in reasoning_text and '180' in reasoning_text:
            reasoning_quality.append("✅ Recognizes triangle angle sum rule")
        
        if 'inscribed' in reasoning_text:
            reasoning_quality.append("✅ Understands inscription geometry")
        
        if 'impossible' in reasoning_text or 'contradiction' in reasoning_text:
            reasoning_quality.append("✅ Identifies logical contradictions")
        
        print(f"\n📈 REASONING QUALITY INDICATORS:")
        for indicator in reasoning_quality:
            print(f"   {indicator}")
        
        if len(reasoning_quality) >= 2:
            print("🎉 AI reasoning shows good geometric understanding!")
        else:
            print("⚠️ AI reasoning could be more detailed")
    
    else:
        print("❌ AI should have detected issues in this complex prompt")

if __name__ == "__main__":
    test_ai_contradiction_detection()
    test_ai_reasoning_quality()
    print("\n🧠 AI-powered contradiction detection testing complete!") 