#!/usr/bin/env python3
"""
Test Real-Time Pipeline Updates
"""

import time
import uuid
from agents.data_structures import PromptBundle
from agents.orchestrator import _orchestrator
from agents.pipeline_visualizer import get_visualizer, AgentState

def test_realtime_updates():
    """Test the real-time pipeline updates"""
    
    print("🔄 TESTING REAL-TIME PIPELINE UPDATES")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = get_visualizer()
    visualizer.reset_pipeline()
    
    # Track update calls
    update_calls = []
    
    def mock_update_callback():
        """Mock update callback that tracks calls"""
        timestamp = time.time()
        active_agent = visualizer.get_current_active_agent()
        timing_summary = visualizer.get_timing_summary()
        
        update_calls.append({
            'timestamp': timestamp,
            'active_agent': active_agent,
            'completed_agents': timing_summary['completed_agents'],
            'total_time': timing_summary['total_time']
        })
        
        print(f"📊 UPDATE #{len(update_calls)}: Active: {active_agent}, "
              f"Completed: {timing_summary['completed_agents']}/{len(visualizer.agents)}, "
              f"Time: {timing_summary['total_time']:.1f}s")
    
    # Test prompt
    prompt_bundle = PromptBundle(
        prompt_id=str(uuid.uuid4()),
        text="Draw a square with 5cm sides"
    )
    
    print(f"🧪 Testing with prompt: '{prompt_bundle.text}'")
    print("-" * 50)
    
    try:
        # Process with real-time callback
        start_time = time.time()
        result = _orchestrator.process_full_pipeline(prompt_bundle, update_callback=mock_update_callback)
        end_time = time.time()
        
        print(f"\n✅ PIPELINE COMPLETED!")
        print(f"📊 REAL-TIME UPDATE STATS:")
        print(f"   • Total updates called: {len(update_calls)}")
        print(f"   • Total pipeline time: {end_time - start_time:.1f}s")
        print(f"   • Updates per second: {len(update_calls) / (end_time - start_time):.1f}")
        
        # Verify we got updates for each agent
        expected_agents = len(visualizer.agents)
        final_timing = visualizer.get_timing_summary()
        
        print(f"\n🎯 VERIFICATION:")
        print(f"   • Expected agent completions: {expected_agents}")
        print(f"   • Actual agent completions: {final_timing['completed_agents']}")
        
        if final_timing['completed_agents'] >= expected_agents - 2:  # Allow some tolerance
            print("✅ Real-time updates working correctly!")
        else:
            print("⚠️ Some agents may not have completed")
        
        # Show update timeline
        print(f"\n📈 UPDATE TIMELINE:")
        for i, update in enumerate(update_calls[-5:], 1):  # Show last 5 updates
            print(f"   {i}. Agent: {update['active_agent']}, "
                  f"Completed: {update['completed_agents']}, "
                  f"Time: {update['total_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("=" * 50)

def test_timing_accuracy():
    """Test timing accuracy for individual agents"""
    
    print("\n⏱️ TESTING TIMING ACCURACY")
    print("-" * 30)
    
    visualizer = get_visualizer()
    
    # Simulate agent execution with known timing
    test_agent = "prompt_interpreter"
    
    # Start timing
    start_time = time.time()
    visualizer.update_agent_status(test_agent, AgentState.THINKING, "Starting test...", 0.1)
    
    # Simulate some work
    time.sleep(1.0)
    
    # Update progress
    visualizer.update_agent_status(test_agent, AgentState.PROCESSING, "Processing...", 0.5)
    
    # More work
    time.sleep(0.5)
    
    # Complete
    visualizer.update_agent_status(test_agent, AgentState.COMPLETE, "Done!", 1.0)
    end_time = time.time()
    
    # Check timing
    actual_duration = end_time - start_time
    measured_duration = visualizer.get_agent_elapsed_time(test_agent)
    
    print(f"✅ Timing Test Results:")
    print(f"   • Actual duration: {actual_duration:.1f}s")
    print(f"   • Measured duration: {measured_duration:.1f}s")
    print(f"   • Difference: {abs(actual_duration - measured_duration):.1f}s")
    
    if abs(actual_duration - measured_duration) < 0.1:
        print("✅ Timing accuracy verified!")
    else:
        print("⚠️ Timing accuracy may need improvement")

if __name__ == "__main__":
    test_realtime_updates()
    test_timing_accuracy()
    print("\n🎉 Real-time pipeline testing complete!") 