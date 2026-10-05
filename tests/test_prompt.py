from core.prompt import generate_tool_schema, build_system_prompt

def dummy_weather(location: str) -> str:
    """Gets the weather for a given location."""
    return f"Sunny in {location}"

def test_generate_tool_schema():
    schema = generate_tool_schema(dummy_weather)
    assert schema["name"] == "dummy_weather"
    assert schema["description"] == "Gets the weather for a given location."
    assert "location" in schema["parameters"]
    assert schema["parameters"]["location"]["type"] == "str"

def test_build_system_prompt():
    prompt = build_system_prompt([dummy_weather])
    assert "You are a helpful AI assistant" in prompt
    assert "dummy_weather" in prompt
    assert "Gets the weather" in prompt
