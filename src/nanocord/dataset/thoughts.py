import re
from string import punctuation


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
    if content:
        thought += f" {content}"
    return thought


def build_json(thought: str) -> str:
    """
    Create a new dataset JSON entry string and,
        return the JSON entry string
    """
    if thought[-1] not in punctuation:
        thought += "."
    # This is a placeholder - the actual JSON structure will be handled by dataset.py
    return thought


import json


def add_to_dataset(thought: str, dataset_file, user_id: str = None):
    """
    Validate a thought, create a dataset JSON entry, and then add it to the dataset
    """
    # We'll validate in the calling function for better control
    entry = {"text": thought}
    dataset_file.write(json.dumps(entry) + "\n")