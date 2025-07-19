#!/usr/bin/env python3
"""
Test script for the AI-powered geometry pipeline.
Tests the orchestrator with text prompts and verifies AI vision capability.
"""

from agents.data_structures import PromptBundle, AgentError
from agents import orchestrator
import uuid


def test_text_only_pipeline():
    """Test the pipeline with text-only prompts."""
    
    print("🧪 Testing AI-Powered Geometry Pipeline")
    print("=" * 60)
    
    test_cases = [
        "Draw a blue triangle with 10 cm sides",
        "Create a red circle with radius 25 pixels", 
        "Make two parallel lines 100 mm apart",
        "Draw a green square with perpendicular lines through the center"
        "Draw a square inside a circle with diameter of 10cm"
    ]
    
    for i, prompt_text in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {prompt_text}")
        print("-" * 40)
        
        try:
            # Create prompt bundle
            prompt_bundle = PromptBundle(
                prompt_id=str(uuid.uuid4()),
                text=prompt_text,
                images=[],
                meta={"source": "test_script"}
            )
            
            # Process with orchestrator
            result = orchestrator.handle(prompt_bundle)
            
            if isinstance(result, AgentError):
                print(f"❌ Error: {result.error}")
                print(f"   Message: {result.message}")
                if result.details:
                    print(f"   Details: {result.details}")
            else:
                print(f"✅ Success!")
                print(f"   Objects: {len(result.objects)}")
                print(f"   Constraints: {len(result.constraints)}")
                print(f"   Status: {result.status.value}")
                
                # Show object details
                for j, obj in enumerate(result.objects, 1):
                    print(f"   Object {j}: {obj.type} (ID: {obj.id})")
                    if obj.properties:
                        key_props = {k: v for k, v in obj.properties.items() if k in ['radius', 'length', 'width', 'height', 'side_length']}
                        if key_props:
                            print(f"      Properties: {key_props}")
                
                # Show constraints
                if result.constraints:
                    for j, constraint in enumerate(result.constraints, 1):
                        print(f"   Constraint {j}: {constraint.type} between {len(constraint.objects)} objects")
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 AI Pipeline Testing Complete!")


def test_api_key_availability():
    """Test if the OpenAI API key is properly configured."""
    
    print("\n🔑 Testing OpenAI API Configuration")
    print("-" * 40)
    
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        print(f"✅ API Key found: {api_key[:20]}...")
        
        # Quick test of OpenAI connection
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Simple test call to verify the key works
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print("✅ OpenAI API connection successful!")
            
        except Exception as e:
            print(f"❌ OpenAI API test failed: {e}")
            
    else:
        print("❌ No API key found in environment")


def show_pipeline_status():
    """Show the current status of all pipeline components."""
    
    print("\n📋 Current Pipeline Component Status")
    print("-" * 40)
    
    components = {
        "Orchestrator": "✅ Ready",
        "Prompt Interpreter": "✅ Ready", 
        "Image Pre-Processor": "✅ Ready",
        "AI Vision Interpreter": "✅ Ready (GPT-4 Vision)",
        "Symbolic Geometry Planner": "⏳ Not implemented",
        "Layout Designer": "⏳ Not implemented", 
        "Renderer": "⏳ Not implemented",
        "Verifier": "⏳ Not implemented",
        "Stylist": "⏳ Not implemented"
    }
    
    for component, status in components.items():
        print(f"   {component}: {status}")
    
    print("\n💡 Ready for:")
    print("   - Text prompt processing")
    print("   - Geometry object extraction") 
    print("   - Constraint detection")
    print("   - AI image analysis (when images provided)")
    print("   - Basic visualization via frontend")


if __name__ == "__main__":
    show_pipeline_status()
    test_api_key_availability()
    test_text_only_pipeline()
    
    print("\n🚀 Next Steps:")
    print("   1. Run: streamlit run frontend.py")
    print("   2. Try text prompts with geometric descriptions")
    print("   3. Upload images to test AI vision capabilities")
    print("   4. Implement remaining pipeline agents for full functionality") 