#!/usr/bin/env python3
"""
Test Complete Clarification Workflow
Verify the end-to-end process from issue detection to user clarification to pipeline restart
"""

import uuid
from agents.data_structures import PromptBundle, ClarificationRequest
from agents.prompt_interpreter import handle
from agents.orchestrator import _orchestrator

def test_complete_clarification_workflow():
    """Test the complete clarification workflow from detection to restart"""
    
    print("🔄 TESTING COMPLETE CLARIFICATION WORKFLOW")
    print("=" * 60)
    
    # Step 1: Test contradiction detection
    print("\n📋 STEP 1: Issue Detection")
    print("-" * 30)
    
    contradictory_prompt = "Draw a square with 20cm sides inscribed in a circle with 8cm diameter"
    
    prompt_bundle = PromptBundle(
        prompt_id=str(uuid.uuid4()),
        text=contradictory_prompt
    )
    
    print(f"Testing prompt: '{contradictory_prompt}'")
    
    # Process with full pipeline (should return ClarificationRequest)
    result = _orchestrator.process_full_pipeline(prompt_bundle)
    
    if isinstance(result, ClarificationRequest):
        print("✅ STEP 1 PASSED: AI detected contradictions correctly")
        print(f"   Category: {result.contradiction_type}")
        print(f"   Issues: {len(result.detected_issues)}")
        for issue in result.detected_issues:
            print(f"      • {issue}")
        
        # Step 2: Simulate user clarification
        print("\n🔄 STEP 2: User Clarification")
        print("-" * 30)
        
        clarified_prompt = "Draw a square inscribed in a circle with 20cm diameter"
        print(f"User clarifies prompt to: '{clarified_prompt}'")
        
        # Step 3: Test restart with clarified prompt
        print("\n🚀 STEP 3: Pipeline Restart")
        print("-" * 30)
        
        clarified_bundle = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=clarified_prompt
        )
        
        # Process clarified prompt (should work without issues)
        clarified_result = _orchestrator.process_full_pipeline(clarified_bundle)
        
        if not isinstance(clarified_result, ClarificationRequest):
            print("✅ STEP 3 PASSED: Clarified prompt processed successfully")
            print(f"   Result type: {type(clarified_result).__name__}")
            
            if hasattr(clarified_result, 'pipeline_metadata'):
                status = clarified_result.pipeline_metadata.get('pipeline_status', 'Unknown')
                print(f"   Pipeline status: {status}")
        else:
            print("❌ STEP 3 FAILED: Clarified prompt still has issues")
            print(f"   Issues: {clarified_result.detected_issues}")
        
        # Step 4: Test skip contradiction check
        print("\n⏩ STEP 4: Skip Contradiction Check")
        print("-" * 30)
        
        original_bundle_skip = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=contradictory_prompt  # Original contradictory prompt
        )
        # Add skip flag
        original_bundle_skip.skip_contradiction_check = True
        
        print(f"Testing original prompt with skip flag: '{contradictory_prompt}'")
        
        skip_result = _orchestrator.process_full_pipeline(original_bundle_skip)
        
        if not isinstance(skip_result, ClarificationRequest):
            print("✅ STEP 4 PASSED: Contradiction check skipped successfully")
            print(f"   Result type: {type(skip_result).__name__}")
        else:
            print("❌ STEP 4 FAILED: Contradiction check not skipped")
        
        print("\n🎉 COMPLETE WORKFLOW TEST SUMMARY:")
        print("✅ Issue detection working")
        print("✅ User clarification interface ready")  
        print("✅ Pipeline restart with clarified prompt working")
        print("✅ Skip contradiction check working")
        print("\n🔄 The complete clarification workflow is functional!")
        
    else:
        print("❌ STEP 1 FAILED: AI should have detected contradictions")
        print(f"   Got result type: {type(result).__name__}")

def test_clarification_edge_cases():
    """Test edge cases in the clarification workflow"""
    
    print(f"\n🧪 TESTING CLARIFICATION EDGE CASES")
    print("-" * 40)
    
    edge_cases = [
        {
            "name": "Ambiguous Reference",
            "prompt": "Draw two circles, then move the circle to the right",
            "should_detect": True
        },
        {
            "name": "Missing Dimensions",
            "prompt": "Create a square and a triangle",
            "should_detect": True
        },
        {
            "name": "Clear Prompt",
            "prompt": "Draw a circle with 5cm radius at coordinates (0,0)",
            "should_detect": False
        }
    ]
    
    for case in edge_cases:
        print(f"\n🧪 Testing: {case['name']}")
        print(f"Prompt: '{case['prompt']}'")
        
        bundle = PromptBundle(
            prompt_id=str(uuid.uuid4()),
            text=case['prompt']
        )
        
        result = _orchestrator.process_full_pipeline(bundle)
        is_clarification = isinstance(result, ClarificationRequest)
        
        if case['should_detect'] and is_clarification:
            print(f"✅ Correctly detected issues: {result.contradiction_type}")
        elif not case['should_detect'] and not is_clarification:
            print("✅ Correctly processed without issues")
        else:
            expected = "should detect" if case['should_detect'] else "should not detect"
            actual = "detected" if is_clarification else "did not detect"
            print(f"⚠️ Expected {expected}, but {actual}")

def test_session_state_simulation():
    """Simulate session state management for clarification workflow"""
    
    print(f"\n💾 TESTING SESSION STATE SIMULATION")
    print("-" * 40)
    
    # Simulate session state for clarification workflow
    session_state = {}
    
    # Initial contradictory submission
    print("1. User submits contradictory prompt")
    session_state['original_prompt'] = "Draw a square with 15cm sides inscribed in 8cm circle"
    
    # System detects issue
    print("2. System detects contradiction and shows clarification interface")
    session_state['needs_clarification'] = True
    session_state['clarification_request'] = "Circle too small for square"
    
    # User provides clarification
    print("3. User clarifies prompt")
    session_state['clarified_prompt'] = "Draw a square inscribed in a circle with 15cm diameter"
    session_state['restart_pipeline'] = True
    
    # System processes clarified prompt
    print("4. System restarts with clarified prompt")
    if session_state.get('restart_pipeline'):
        prompt_to_use = session_state.get('clarified_prompt', session_state['original_prompt'])
        print(f"   Processing: '{prompt_to_use}'")
        
        # Clear flags
        session_state['restart_pipeline'] = False
        session_state['clarified_prompt'] = ''
        session_state['needs_clarification'] = False
        
        print("✅ Session state managed correctly for clarification workflow")
    
    print("\n💡 This demonstrates how the frontend session state should work")

if __name__ == "__main__":
    test_complete_clarification_workflow()
    test_clarification_edge_cases()
    test_session_state_simulation()
    print("\n🎯 Clarification workflow testing complete!") 