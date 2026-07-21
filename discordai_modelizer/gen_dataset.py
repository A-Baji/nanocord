from appdirs import user_data_dir
from json import load, dumps
from datetime import timedelta
from dateutil import parser
from string import punctuation
from os import path
import re
import pathlib


class UserNotFoundError(Exception):
    pass


def parse_logs(
    file: str,
    channel: str,
    user: str,
    thought_time=5,
    thought_max: int = None,
    thought_min=6,
):

    def validate_thought(thought: str) -> bool:
        """
        If the thought's word count is within `thought_min` and `thought_max`,
            return True
        """
        word_count = len(thought.split(" ")) - 1
        if word_count >= thought_min and thought_max >= word_count:
            return True

    def cleanup_string(msg: str) -> str:
        """
        Remove URLs and slurs from a string and,
            return the string
        """
        hate_speech_words = ["nigg", "fag", "gay", "tard"]

        def censor_hate(match):
            word = match.group()
            # Find all vowels and replace them along with the next two characters
            censored_word = re.sub(
                r"([aeiou]).{0,2}",
                lambda m: "*" * len(m.group()),
                word,
                flags=re.IGNORECASE,
            )
            return censored_word

        url_pattern = re.compile(r"\bhttps?://\S+|\bftp://\S+|\bfile://\S+")
        msg = url_pattern.sub("", msg)

        for word in hate_speech_words:
            pattern = re.compile(rf"(\b{re.escape(word)}\w{{0,1}})", re.IGNORECASE)
            msg = pattern.sub(censor_hate, msg)

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
        return dumps({"prompt": f"{user[:13]} says:", "completion": thought}) + "\n"

    def add_to_dataset(thought: str):
        """
        Validate a thought, create a dataset JSON entry, and then add it to the dataset
        """
        if validate_thought(thought):
            dataset.write(build_json(cleanup_string(thought)))

    files_path = pathlib.Path(user_data_dir(appname="discordai"))
    dataset = open(files_path / f"{user[:13]}_{channel[:4]}_data_set.jsonl", "w")
    thought_max = 999999 if not thought_max else thought_max
    if "#" in user:
        username, user_id = user.split("#")
    else:
        username, user_id = user, None
    with open(file, "r", encoding="utf-8") as data_file:
        data = load(data_file)
        messages = [
            msg
            for msg in data["messages"]
            if msg["author"].get("name") == username
            and (user_id is None or msg["author"].get("discriminator") == user_id)
        ]
        if not messages:
            raise UserNotFoundError(
                f"No messages found in chat logs for user: {username}"
            )
        thought = build_thought("", messages[0])
        for i, msg in enumerate(messages[1::]):
            if msg["content"]:
                prev_timestamp = parser.parse(messages[i]["timestamp"])
                curr_timestamp = parser.parse(msg["timestamp"])
                differentiation = (curr_timestamp - prev_timestamp) / timedelta(
                    milliseconds=1
                )
                if differentiation > thought_time * 1000:
                    add_to_dataset(thought)
                    thought = build_thought("", msg)
                else:
                    thought = build_thought(thought, msg)
        add_to_dataset(thought)
    dataset.close()
    if path.getsize(files_path / f"{user[:13]}_{channel[:4]}_data_set.jsonl") == 0:
        print(
            "WARNING: The resulting dataset is empty. Please double check your parameters."
        )


def get_lines(file_name: str, N=1000, offset=0, distributed=False, reverse=False):
    with open(file_name, "r") as f:
        lines = f.readlines()
    f.close()

    num_lines = len(lines)

    if distributed:
        step = (num_lines - offset) // N
    else:
        step = 1

    if reverse:
        lines = lines[::-1]

    selected_lines = lines[offset:][:: step or 1][:N]

    with open(file_name, "w") as f:
        f.writelines(selected_lines)
    f.close()
