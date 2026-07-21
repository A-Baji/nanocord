def set_bot_key_help_str(is_parent=False):
    return f"{'in your config' if is_parent else 'as the DISCORD_BOT_TOKEN environment variable'}"


def setup_model_create(model_subcommand, is_parent=False):
    model_create = model_subcommand.add_parser(
        "create",
        help="Download Discord channel logs, parse them into a dataset, and save the results locally",
    )
    model_create_required_named = model_create.add_argument_group(
        "required named arguments"
    )
    model_create_optional_named = model_create.add_argument_group(
        "optional named arguments"
    )

    model_create_required_named.add_argument(
        "-d",
        "--discord-token",
        type=str,
        dest="discord_token",
        help=f"The Discord token for your bot. Must either be passed in as an argument or set {set_bot_key_help_str(is_parent)}",
    )
    model_create_required_named.add_argument(
        "-c",
        "--channel",
        required=True,
        type=str,
        dest="channel",
        help="The ID of the Discord channel you want to use",
    )
    model_create_required_named.add_argument(
        "-u",
        "--user",
        required=True,
        type=str,
        dest="user",
        help="The unique username of the Discord user you want to use",
    )

    model_create_optional_named.add_argument(
        "--ttime",
        "--thought-time",
        type=int,
        default=5,
        required=False,
        dest="thought_time",
        help='The maximum amount of time in seconds to consider two individual messages to be part of the same "thought": DEFAULT=10',
    )
    model_create_optional_named.add_argument(
        "--tmax",
        "--thought-max",
        type=int,
        default=None,
        required=False,
        dest="thought_max",
        help="The maximum length in words of each thought: DEFAULT=None",
    )
    model_create_optional_named.add_argument(
        "--tmin",
        "--thought-min",
        type=int,
        default=6,
        required=False,
        dest="thought_min",
        help="The minimum length in words of each thought: DEFAULT=4",
    )
    model_create_optional_named.add_argument(
        "-m",
        "--max-entries",
        type=int,
        default=1000,
        required=False,
        dest="max_entries",
        help="The max amount of entries (by lines) that may exist in the dataset: DEFAULT=1000",
    )
    model_create_optional_named.add_argument(
        "--os",
        "--offset",
        type=int,
        default=0,
        required=False,
        dest="offset",
        help="The offset by line index starting at 0 for where to start selecting lines for the dataset: DEFAULT=0",
    )
    model_create_optional_named.add_argument(
        "--distributed",
        action="store_true",
        required=False,
        dest="distributed",
        help="Select lines as an even distribution instead of sequentially",
    )
    model_create_optional_named.add_argument(
        "--reverse_lines",
        action="store_true",
        required=False,
        dest="reverse",
        help="Reverse the order in which to select lines for the dataset",
    )
    model_create_optional_named.add_argument(
        "--dirty",
        action="store_false",
        required=False,
        dest="dirty",
        help="Skip the clean up step for outputted files",
    )
    model_create_optional_named.add_argument(
        "--redownload",
        action="store_true",
        required=False,
        dest="redownload",
        help="Redownload the Discord chat logs",
    )
    model_create_optional_named.add_argument(
        "--use_existing",
        action="store_true",
        required=False,
        dest="use_existing",
        help="Use an existing dataset that may have been manually revised",
    )
