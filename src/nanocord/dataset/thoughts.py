import re
import emoji
from datetime import datetime
from datetime import timedelta
from typing import Tuple, List, Dict


class UserNotFoundError(Exception):
    pass


def normalize_mentions(msg: dict) -> Tuple[str, List[str]]:
    """
    Normalize mentions in a Discord message content by:
    1. Swapping nickname-based mentions for username-based mentions (for stored content)
    2. Identifying real mentions that should be excluded from word counting
    3. Filtering out reply-type false positives (mentions that appear in the reply array but
       were never literally @-tagged in the content)

    For each entry in msg.get("mentions", []):
      - candidate = "@" + (entry.get("nickname") or entry["name"])
      - If candidate is NOT a substring of msg["content"]: skip this entry entirely
        (this is the reply-false-positive case from point 3 - the mention array lists
        the replied-to user but they were never literally @-tagged in the text, so there
        is nothing to swap or count).
      - Otherwise it's a real mention: if entry.get("nickname") is truthy,
        replace that exact "@nickname" substring in the content with "@" + entry["name"]
        (do this on a working copy of the content string, not msg["content"] directly,
        since multiple mentions in one message must all be applied before returning).
      - Whether or not a swap was needed, record "@" + entry["name"] (the canonical
        post-swap form) as a real mention for this message.

    Args:
        msg: A Discord message dictionary from DiscordChatExporter

    Returns:
        Tuple of (normalized_content, real_mentions) where:
        - normalized_content is the message content with nickname mentions swapped for usernames
        - real_mentions is a list of username-based mention strings that were actually found in content
    """
    # Start with the original content
    content = msg.get("content", "")

    # If there are no mentions, return the content unchanged and empty mentions list
    mentions = msg.get("mentions", [])
    if not mentions:
        return (content, [])

    # Track real mentions that were found in content
    real_mentions = []

    # Create a working copy of the content to modify - we'll check against the original
    # content for false positives, but modify a copy for replacements
    working_content = content

    for entry in mentions:
        # The candidate mention string is what was originally rendered in content
        # This should be either "@nickname" if nickname exists, or "@name" if not
        if entry.get("nickname"):
            candidate = "@" + entry["nickname"]
        else:
            candidate = "@" + entry["name"]

        # If this mention is not literally in the original content, it's a false positive
        # from the reply array and should be ignored entirely
        if candidate not in content:
            continue

        # This is a real mention - record the username-based version for counting purposes
        real_mentions.append("@" + entry["name"])

        # If there's a nickname, swap it for the username in the content
        if entry.get("nickname"):
            # Replace only the first occurrence to avoid replacing one mention with another
            working_content = working_content.replace(candidate, "@" + entry["name"], 1)

    return (working_content, real_mentions)


def strip_emoji_and_emotes(text: str, mentions: list = None, emote_codes: list = None) -> str:
    """
    Remove emoji and Discord custom emote shortcodes from text for word-counting purposes only.

    This function is for validation logic only - the actual stored thought text
    (used for training) must keep emoji and emotes exactly as-is, unchanged.

    Args:
        text: Input text that may contain unicode emoji and :emote_name: shortcodes
        mentions: List of mention strings to remove from text (e.g., ["@user1", "@user2"])
        emote_codes: List of emote codes to remove from text (e.g., ["thumbsup", "smile"])

    Returns:
        Text with all emoji, emote shortcodes, and mentions removed, suitable only for word counting
    """
    # Remove mentions first
    if mentions:
        for mention in mentions:
            text = text.replace(mention, "", 1)

    # Remove emote codes next
    if emote_codes:
        for code in emote_codes:
            text = text.replace(f":{code}:", "", 1)

    # Remove unicode emoji
    text_without_emoji = emoji.replace_emoji(text, replace="")
    return text_without_emoji


def validate_thought(thought: str, thought_min: int = 6, thought_max: int = None, mentions: list = None, emote_codes: list = None) -> bool:
    """
    If the thought's word count is within `thought_min` and `thought_max`,
        return True

    Note: Emoji and Discord custom emote shortcodes (like :thumbsup:) do not
    count toward the word threshold. A short real-text message padded with
    several emoji no longer incorrectly passes thought_min.
    """
    # Count words (excluding empty strings) - using stripped text for counting
    word_count = len([word for word in strip_emoji_and_emotes(thought, mentions, emote_codes).split() if word])
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
            "mentions": list[str],             # list of mention strings (usernames) that were
                                                 # found in any message in this thought
            "emote_codes": list[str],          # list of emote codes that were found in any
                                                 # message in this thought
          }
    """
    if not messages:
        return []

    def start_new_thought(msg: dict) -> dict:
        # Normalize mentions first, then clean content
        normalized_content, real_mentions = normalize_mentions(msg)
        cleaned_content = cleanup_string(normalized_content) if normalized_content else normalized_content
        msg = {**msg, "content": cleaned_content}

        # Extract emote codes from inlineEmojis
        emote_codes = []
        for emoji_entry in msg.get("inlineEmojis", []):
            emote_codes.append(emoji_entry["code"])

        return {
            "text": build_thought("", msg),
            "message_ids": [msg.get("id")],
            "reply_reference_id": msg["reference"]["messageId"] if "reference" in msg else None,
            "mentions": real_mentions,  # Initialize with the mentions from this message
            "emote_codes": emote_codes,  # Initialize with the emote codes from this message
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
            # Normalize mentions first, then clean content
            normalized_content, real_mentions = normalize_mentions(msg)
            cleaned_content = cleanup_string(normalized_content) if normalized_content else normalized_content
            cleaned_msg = {**msg, "content": cleaned_content}

            current["text"] = build_thought(current["text"], cleaned_msg)
            current["message_ids"].append(msg.get("id"))

            # Accumulate mentions from this message into the thought's mentions list
            current["mentions"].extend(real_mentions)

            # Accumulate emote codes from this message into the thought's emote_codes list
            for emoji_entry in msg.get("inlineEmojis", []):
                emote_code = emoji_entry["code"]
                if emote_code not in current["emote_codes"]:
                    current["emote_codes"].append(emote_code)

    thoughts.append(current)
    return thoughts