import json


def extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if "```json\n" in content:
        start = content.find("```json\n") + 8
        end = content.find("\n```", start)
        json_str = content[start:end].strip() if end != -1 else content[start:].strip()
    elif "```\n" in content:
        start = content.find("```\n") + 4
        end = content.find("\n```", start)
        json_str = content[start:end].strip() if end != -1 else content[start:].strip()
    else:
        json_str = content.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Error parsing JSON: {json_str}")
        return None
