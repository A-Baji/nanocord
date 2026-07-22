#!/usr/bin/env python3

import os

# Test setting and retrieving environment variable
print("Testing environment variable handling...")

# Set test token
test_token = "test_bot_token_12345"
os.environ["DISCORD_BOT_TOKEN"] = test_token

# Check if it was set correctly
retrieved_token = os.environ.get("DISCORD_BOT_TOKEN")
print(f"Set token: {test_token}")
print(f"Retrieved token: {retrieved_token}")
print(f"Tokens match: {test_token == retrieved_token}")

# Test with None value
os.environ["DISCORD_BOT_TOKEN"] = ""
retrieved_token = os.environ.get("DISCORD_BOT_TOKEN")
print(f"Empty token retrieved: '{retrieved_token}'")

# Test with non-existent key
missing_token = os.environ.get("NON_EXISTENT_TOKEN", "default_value")
print(f"Missing token with default: {missing_token}")