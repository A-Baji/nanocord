# DiscordAI Modelizer
DiscordAI Modelizer is a python package that can generate custom openai models based on a Discord user's chat history in a Discord channel. It uses [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to download the logs of a channel, processes the logs into a usable dataset, and then uses [OpenAI's API](https://beta.openai.com/docs/introduction) to create a customized model. It also wraps some of the tools from the OpenAI API to help with managing customizations.

DiscordAI Modelizer is primarily used as a subcomponent of [DiscordAI](https://github.com/A-Baji/discordAI), but may also be used independently.

## Installation
`pip install -U git+https://github.com/A-Baji/discordAI-modelizer.git`
### Or
1. Download/clone the source locally
2. Run `pip install -U <path to source>/.`
3. The source may now be deleted

## Commands
### Model
Commands related to your OpenAI models
#### `discordai_modelizer model list`
* List your OpenAI models
#### `discordai_modelizer model create`
* Create a new OpenAI customized model by downloading the specified chat logs, parsing them into a usable dataset, and then training a customized model using openai
* For proper usage, see the [DiscordAI guide](https://github.com/A-Baji/discordAI#create-a-new-customized-openai-model)
#### `discordai_modelizer model delete`
* Delete an OpenAI model
### Job
Commands related to your OpenAI jobs
#### `discordai_modelizer job list`
* List your OpenAI customization jobs
#### `discordai_modelizer job info`
* Get an OpenAI customization job's info
#### `discordai_modelizer job event`
* Get an OpenAI customization job's events
#### `discordai_modelizer job cancel`
* Cancel an OpenAI customization job

## Disclaimer
This application allows users to download the chat history of any channel for which they have permission to invite a bot, and then use those logs to create an openai model based on a user's chat messages. It is important to note that this application should only be used with the consent of all members of the channel. Using this application for malicious purposes, such as impersonation, or without the consent of all members is strictly prohibited.

By using this application, you agree to use it responsibly. The developers of this application are not responsible for any improper use of the application or any consequences resulting from such use. We strongly discourage using this application for any unethical purposes.

This application is not affiliated with or endorsed by Discord, Inc. The use of the term "Discord" in our product name is solely for descriptive purposes to indicate compatibility with the Discord platform.