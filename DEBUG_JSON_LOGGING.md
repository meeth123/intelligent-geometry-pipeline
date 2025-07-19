# 🔍 JSON Response Logging System

## Overview

Every AI agent now automatically logs all JSON responses during runtime for debugging purposes. This helps identify and fix JSON parsing issues across the entire pipeline.

## How It Works

### Automatic Logging
- **All agents** now save JSON responses to `debug_json_responses/` directory
- **Real-time logging** during pipeline execution
- **Multiple status types** captured for each response

### File Structure
```
debug_json_responses/
├── prompt_interpreter_[id]_[timestamp]_raw_response.json
├── prompt_interpreter_[id]_[timestamp]_parsed_success.json
├── layout_designer_[id]_[timestamp]_raw_response.json
├── layout_designer_[id]_[timestamp]_parse_failed.json
├── renderer_[id]_[timestamp]_cleanup_failed.json
└── ...
```

### Status Types
- `raw_response` - Original AI response from Gemini
- `parsed_success` - Successfully parsed JSON
- `parse_failed` - JSON parsing failed
- `cleaned_success` - Cleaned JSON worked
- `cleanup_failed` - All parsing attempts failed
- `fallback_parsed` - Used fallback parsing

## JSON File Contents

Each debug file contains:
```json
{
  "agent": "agent_name",
  "prompt_id": "unique_identifier", 
  "timestamp": "YYYYMMDD_HHMMSS",
  "status": "response_status",
  "raw_response": "complete_ai_response",
  "response_length": 1234
}
```

## Debugging Workflow

### 1. Run Your Pipeline
```python
from agents.orchestrator import _orchestrator
from agents.data_structures import PromptBundle

prompt_bundle = PromptBundle(
    prompt_id=str(uuid.uuid4()),
    text="Your geometry prompt here"
)

result = _orchestrator.process_full_pipeline(prompt_bundle)
```

### 2. Check Debug Files
```bash
ls -la debug_json_responses/
```

### 3. Identify Issues
Look for files with status:
- `parse_failed` - JSON parsing errors
- `cleanup_failed` - Enhanced cleaning couldn't fix it
- `fallback_parsed` - Agent used fallback instead of AI response

### 4. Inspect Problematic Responses
```python
import json

with open('debug_json_responses/agent_id_timestamp_parse_failed.json', 'r') as f:
    debug_data = json.load(f)
    print(debug_data['raw_response'])  # See exact problematic JSON
```

## Benefits

✅ **Immediate Problem Identification**: See exactly which agent failed and why  
✅ **Complete Response Capture**: Full AI responses saved for analysis  
✅ **Status Tracking**: Know if issue was parsing, cleaning, or fallback  
✅ **Timeline Tracking**: Timestamps show when issues occurred  
✅ **Pattern Recognition**: Identify recurring JSON formatting problems  

## Agent Coverage

All agents now have JSON logging:
- ✅ Prompt Interpreter
- ✅ Symbolic Geometry Planner  
- ✅ Layout Designer
- ✅ Renderer
- ✅ Math Consistency Verifier

## Example Usage

```bash
# Find all failed JSON responses
find debug_json_responses/ -name "*failed*"

# Count responses by agent
ls debug_json_responses/ | cut -d'_' -f1 | sort | uniq -c

# Find latest responses
ls -lt debug_json_responses/ | head -10
```

## Tips

💡 **Check after pipeline failures**: Failed responses help identify root causes  
💡 **Compare successful vs failed**: See what makes JSON parsing work vs fail  
💡 **Monitor file sizes**: Very large responses might need token limits  
💡 **Clean periodically**: Debug files can accumulate over time  

---

This logging system makes debugging JSON parsing issues **10x easier** by providing complete visibility into AI responses! 🎯 