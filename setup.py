import platform
import setuptools
import pathlib
import sys
import discordai_modelizer as package

min_py_version = (3, 9)

if sys.version_info < min_py_version:
    sys.exit(
        "DiscordAI Modelizer is only supported for Python {}.{} or higher".format(
            *min_py_version
        )
    )

here = pathlib.Path(__file__).parent.resolve()
with open(pathlib.Path(here, "requirements.txt")) as f:
    requirements = [r for r in f.read().splitlines()]


def get_dce_path():
    os_name = platform.system()

    if os_name == "Linux":
        return "DiscordChatExporter.Cli.linux-x64"
    elif os_name == "Darwin":
        return "DiscordChatExporter.Cli.osx-x64"
    elif os_name == "Windows":
        return "DiscordChatExporter.Cli.win-x64"


setuptools.setup(
    name=package.__name__,
    version=package.__version__,
    author="Adib Baji",
    author_email="bidabaji@gmail.com",
    description="A package that utilizes OpenAI to create custom AI models out of your Discord chat history",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/A-Baji/discordAI-modelizer",
    packages=setuptools.find_packages(),
    package_dir={"": "."},
    package_data={"": [f"DiscordChatExporter/{get_dce_path()}/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            f"{package.__name__}={package.__name__}.command_line.command_line:{package.__name__}"
        ],
    },
    install_requires=requirements,
    python_requires="~={}.{}".format(*min_py_version),
)
