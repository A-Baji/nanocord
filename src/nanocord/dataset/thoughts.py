import re
from datetime import datetime
from datetime import timedelta


class UserNotFoundError(Exception):
    pass


def validate_thought(thought: str, thought_min: int = 6, thought_max: int = None) -> bool:
    """
    If the thought's word count is within `thought_min` and `thought_max`,
        return True
    """
    # Count words (excluding empty strings)
    word_count = len([word for word in thought.split() if word])
    if thought_max is None:
        thought_max = 999999
    if word_count >= thought_min and thought_max >= word_count:
        return True
    return False


def cleanup_string(msg: str) -> str:
    """
    Remove URLs from a string and,
        return the string
    """
    url_pattern = re.compile(r"\bhttps?://\S+|\bftp://\S+|\bfile://\S+")
    msg = url_pattern.sub("", msg)

    return msg


def build_thought(thought: str, msg: dict) -> str:
    """
    Add a message to a thought and,
        return the thought
    """
    content = msg["content"].strip()  # Remove leading/trailing spaces
    if not content:
        return thought
    return f"{thought} {content}".strip()


def group_into_thoughts(messages: list, thought_time: int) -> list:
    """
    Group a chronologically-ordered list of Discord messages into "thoughts" -
    runs of consecutive messages considered part of the same idea.

    A new thought starts whenever any of the following is true for the
    current message:
      - it is the first message in `messages`
      - more than `thought_time` seconds have passed since the previous message
      - the message is itself a reply (has a "reference" key) - a reply always
        starts a new thought and is always that thought's first message,
        regardless of the time gap from the previous message

    Args:
        messages: Chronologically-ordered list of Discord message dicts, as
                  parsed from a DiscordChatExporter export
        thought_time: Maximum seconds between two messages to be considered
                      part of the same thought

    Returns:
        List of thought dicts, each shaped:
          {
            "text": str,                       # concatenated cleaned message content
            "message_ids": list[str],          # every message id in this thought, in order
            "reply_reference_id": str | None,  # if the thought's first message was a
                                                 # reply, msg["reference"]["messageId"];
                                                 # otherwise None
          }
    """
    if not messages:
        return []

    def start_new_thought(msg: dict) -> dict:
        content = cleanup_string(msg["content"]) if msg["content"] else msg["content"]
        msg = {**msg, "content": content}
        return {
            "text": build_thought("", msg),
            "message_ids": [msg.get("id")],
            "reply_reference_id": msg["reference"]["messageId"] if "reference" in msg else None,
        }

    thoughts = []
    current = start_new_thought(messages[0])

    for i in range(1, len(messages)):
        prev_msg = messages[i - 1]
        msg = messages[i]
        is_reply = "reference" in msg

        prev_timestamp = datetime.fromisoformat(prev_msg["timestamp"])
        curr_timestamp = datetime.fromisoformat(msg["timestamp"])
        gap_ms = (curr_timestamp - prev_timestamp) / timedelta(milliseconds=1)

        if is_reply or gap_ms > thought_time * 1000:
            thoughts.append(current)
            current = start_new_thought(msg)
        else:
            content = cleanup_string(msg["content"]) if msg["content"] else msg["content"]
            cleaned_msg = {**msg, "content": content}
            current["text"] = build_thought(current["text"], cleaned_msg)
            current["message_ids"].append(msg.get("id"))

    thoughts.append(current)
    return thoughts