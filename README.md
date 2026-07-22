# NanoCord

## Installation

```bash
pip install nanocord
```

## Usage

### Prerequisites

NanoCord uses DiscordChatExporter to download Discord channel exports before it can build a dataset. It is now treated as an external prerequisite rather than a bundled dependency.

1. Download the latest DiscordChatExporter release from https://github.com/Tyrrrz/DiscordChatExporter/releases.
2. Install it and make sure the executable is available on your machine.
3. If the tool is not detected automatically, set the DISCORD_CHAT_EXPORTER_PATH environment variable to the full path of the executable, or enter the path when prompted by the CLI.

Example:

- PowerShell: $env:DISCORD_CHAT_EXPORTER_PATH = "C:\\Path\\To\\DiscordChatExporter.Cli.exe"
- Bash: export DISCORD_CHAT_EXPORTER_PATH="/path/to/DiscordChatExporter.Cli"

### Commands

```bash
# Create a dataset from Discord logs
nanocord dataset create -c <channel_id> -u <user_id> -d <discord_bot_token>
```
