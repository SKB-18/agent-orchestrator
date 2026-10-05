import pytest
from core.parser import parse_tool_call

def test_parse_clean_json():
    text = '{"tool_name": "get_weather", "kwargs": {"location": "London"}}'
    result = parse_tool_call(text)
    assert result["tool_name"] == "get_weather"
    assert result["kwargs"]["location"] == "London"

def test_parse_markdown_json():
    text = '''Here is the tool call:
```json
{"tool_name": "calculator", "kwargs": {"expression": "2 + 2"}}
```
I hope this helps!'''
    result = parse_tool_call(text)
    assert result["tool_name"] == "calculator"
    assert result["kwargs"]["expression"] == "2 + 2"

def test_parse_invalid_json():
    text = "I don't know how to do that."
    with pytest.raises(ValueError, match="No valid JSON object found"):
        parse_tool_call(text)

def test_parse_missing_required_keys():
    text = '{"tool": "get_weather"}'
    with pytest.raises(ValueError, match="Missing required keys in tool call"):
        parse_tool_call(text)
