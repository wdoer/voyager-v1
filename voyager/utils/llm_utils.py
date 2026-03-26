def extract_text_from_ai_message(message_content):
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        text = ""
        for block in message_content:
            if isinstance(block, dict) and "text" in block:
                text += block["text"]
            elif isinstance(block, str):
                text += block
        return text
    return str(message_content)
