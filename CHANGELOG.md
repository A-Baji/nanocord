# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [3.0.11] - 07-16-2024

### Fixed

- remove `.identifier` files

## [3.0.10] - 07-16-2024

### Fixed

- log downloading support for all operating systems

## [3.0.9] - 07-11-2024

### Changed

- consistently capitalize Discord and OpenAI across the project

## [3.0.8] - 07-11-2024

### Changed

- changed distributed selection mode to a flag: `--distributed`
- fix a small bug for distributed mode with offsets  

## [3.0.7] - 07-11-2024

### Changed

- updated hate speech censoring
- updated file model tagging

## [3.0.6] - 07-10-2024

### Changed

- Moved tests back out of the package

## [3.0.5] - 07-9-2024

### Added

- `--force` arg to model deletion
- error handling for when user was not found in logs

### Changed

- Include `tests/` within the package to allow reuse in parent package

## [3.0.4] - 07-1-2024

### Removed

- deleted unnecessary DiscordChatExporter files

## [3.0.3] - 07-1-2024

### Changed

- update env value handling
- make command help strs dynamic
- reorganize cli to allow reuse in parent

## [3.0.2] - 06-29-2024

### Added

- unit tests

### Changed

- model deletion returns valid JSON on permission error

### Fixed

- multiple bugs for dataset generation, model creation, and cli

## [3.0.1] - 06-27-2024

### Changed

- improve cli argument handling

## [3.0.0] - 06-26-2024

### Added

- support for python 3.12

### Changed

- update `DiscordChatExporter` to v2.43.3
- update and pinned `openai` to v1.35.5
- update reduction method to use an offset, a selection mode of either sequential or distributed, and a flag to iterate in reverse to reduce the line selection
- update some default CLI command argument values
- update some CLI help descriptions
- major refactors

### Removed

- support for python 3.8

## [2.0.1] - 06-15-2023

### Changed

- update cli user argument description 

## [2.0.0] - 06-15-2023

### Added

- a changelog
- confirmation dialogue for model deletion

### Changed

- flag to use an existing dataset for model creation to allow manual revision
- user input for model creation updated to work with new Discord naming system
- tweaked readme

## [1.2.2] - 02-22-2023

### Fixed

- bug with end of thought punctuation

## [1.2.1] - 02-19-2023

### Fixed

- bug with min and max thought length

## [1.2.0] - 02-17-2023

### Added

- events flag for job status command
- min and max thought length parameters for model creation

## [1.1.0] - 02-11-2023

### Changed

- replaced `subprocess` instances with appropriate OpenAI python API methods

### Removed

- dataset clean up step with OpenAI fine tune API

## [1.0.1] - 02-11-2023

### Changed

- switched to `pathlib` for file path parsing

[3.0.11]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.10...3.0.11
[3.0.10]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.9...3.0.10
[3.0.9]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.8...3.0.9
[3.0.8]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.7...3.0.8
[3.0.7]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.6...3.0.7
[3.0.6]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.5...3.0.6
[3.0.5]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.4...3.0.5
[3.0.4]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.3...3.0.4
[3.0.3]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.2...3.0.3
[3.0.2]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.1...3.0.2
[3.0.1]: https://github.com/A-Baji/discordAI-modelizer/compare/3.0.0...3.0.1
[3.0.0]: https://github.com/A-Baji/discordAI-modelizer/compare/2.0.1...3.0.0
[2.0.1]: https://github.com/A-Baji/discordAI-modelizer/compare/1.2.2...2.0.1
[2.0.0]: https://github.com/A-Baji/discordAI-modelizer/compare/1.2.2...2.0.0
[1.2.2]: https://github.com/A-Baji/discordAI-modelizer/compare/1.2.1...1.2.2
[1.2.1]: https://github.com/A-Baji/discordAI-modelizer/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/A-Baji/discordAI-modelizer/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/A-Baji/discordAI-modelizer/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/A-Baji/discordAI-modelizer/compare/1.0.0...1.0.1