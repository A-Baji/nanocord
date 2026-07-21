def set_openai_help_str(is_parent=False):
    return f"{'in your config' if is_parent else 'as the OPENAI_API_KEY environment variable'}"


def set_bot_key_help_str(is_parent=False):
    return f"{'in your config' if is_parent else 'as the DISCORD_BOT_TOKEN environment variable'}"


def setup_model_list(model_subcommand, is_parent=False):
    model_list = model_subcommand.add_parser(
        "list", help="List your OpenAI customized models"
    )
    model_list_required_named = model_list.add_argument_group(
        "required named arguments"
    )
    model_list_optional_named = model_list.add_argument_group(
        "optional named arguments"
    )

    model_list_required_named.add_argument(
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key to list the models for. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
    )
    model_list_optional_named.add_argument(
        "--full",
        action="store_true",
        required=False,
        dest="full",
        help="Return the full details of all the models",
    )


def setup_model_create(model_subcommand, is_parent=False):
    model_create = model_subcommand.add_parser(
        "create",
        help="Create a new OpenAI customized model by downloading the specified chat logs, parsing them into a usable dataset, and then training a customized model using openai",
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
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key to use to create the model. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
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
        "-b",
        "--base-model",
        choices=["davinci", "babbage", "none"],
        default="none",
        required=False,
        dest="base_model",
        help="The base model to use for customization, where `davinci` is the more advanced model and is recommended, while `babbage` is a simpler model and should be reserved for testing since it is cheaper, and `none`, skips the training step: DEFAULT=none",
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


def setup_model_delete(model_subcommand, is_parent=False):
    model_delete = model_subcommand.add_parser(
        "delete",
        help="Delete an OpenAI customized model",
    )
    model_delete_required_named = model_delete.add_argument_group(
        "required named arguments"
    )
    model_delete_optional_named = model_delete.add_argument_group(
        "optional named arguments"
    )

    model_delete_required_named.add_argument(
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key associated with the model to delete. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
    )
    model_delete_required_named.add_argument(
        "-m",
        "--model-id",
        required=True,
        type=str,
        dest="model_id",
        help="Target model id",
    )

    model_delete_optional_named.add_argument(
        "--force",
        action="store_true",
        required=False,
        dest="force",
        help="Skips the deletion confirmation dialogue",
    )


def setup_job_list(job_subcommand, is_parent=False):
    job_list = job_subcommand.add_parser(
        "list", help="List your OpenAI customization jobs"
    )
    job_list_required_named = job_list.add_argument_group("required named arguments")
    job_list_optional_named = job_list.add_argument_group("optional named arguments")

    job_list_required_named.add_argument(
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key to list the jobs for. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
    )
    job_list_optional_named.add_argument(
        "--full",
        action="store_true",
        required=False,
        dest="full",
        help="Return the full details of all the jobs",
    )


def setup_job_info(job_subcommand, is_parent=False):
    job_info = job_subcommand.add_parser(
        "info", help="Get an OpenAI customization job's info"
    )
    job_info_required_named = job_info.add_argument_group("required named arguments")

    job_info_required_named.add_argument(
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key associated with the job to see the info for. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
    )
    job_info_required_named.add_argument(
        "-j",
        "--job-id",
        required=True,
        type=str,
        dest="job_id",
        help="Target job id",
    )


def setup_job_events(job_subcommand, is_parent=False):
    job_events = job_subcommand.add_parser(
        "events", help="Get an OpenAI customization job's events"
    )
    job_events_required_named = job_events.add_argument_group(
        "required named arguments"
    )

    job_events_required_named.add_argument(
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key associated with the job to see the events for. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
    )
    job_events_required_named.add_argument(
        "-j",
        "--job-id",
        required=True,
        type=str,
        dest="job_id",
        help="Target job id",
    )


def setup_job_cancel(job_subcommand, is_parent=False):
    job_cancel = job_subcommand.add_parser(
        "cancel", help="Cancel an OpenAI customization job"
    )
    job_cancel_required_named = job_cancel.add_argument_group(
        "required named arguments"
    )

    job_cancel_required_named.add_argument(
        "-o",
        "--openai-key",
        type=str,
        dest="openai_key",
        help=f"The OpenAI API key associated with the job to cancel. Must either be passed in as an argument or set {set_openai_help_str(is_parent)}",
    )
    job_cancel_required_named.add_argument(
        "-j",
        "--job-id",
        required=True,
        type=str,
        dest="job_id",
        help="Target job id",
    )
