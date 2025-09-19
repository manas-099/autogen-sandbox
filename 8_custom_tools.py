
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
import os
import asyncio
load_dotenv()
gemini_api_key=os.getenv('GEMINI_API_KEY')
print("Gemini key loaded:", gemini_api_key is not None)

async def add_numbers(a: int, b: int) -> int:
    """Adds two numbers and returns the result."""
    return a + b
addition_tool=FunctionTool(add_numbers,description="useful for addition operations.")
async def main():
    cancellation_token = CancellationToken()
    
    # Call the tool properly
    result = await addition_tool.run_json(
        {"a": 10, "b": 20},  # Pass numbers as int, not strings
        cancellation_token
    )

    # Print the result
    print("Tool Result:", addition_tool.return_value_as_string(result))
    print("schema:",addition_tool.schema)

if __name__ == "__main__":
    asyncio.run(main())
