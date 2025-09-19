
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langchain_community.tools import DuckDuckGoSearchResults
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.tools.langchain import LangChainToolAdapter
from dotenv import load_dotenv
from autogen_agentchat.ui import Console

import os
import asyncio
load_dotenv()
gemini_api_key=os.getenv('GEMINI_API_KEY')
print("Gemini key loaded:", gemini_api_key is not None)
search_tool = DuckDuckGoSearchResults()
tool = LangChainToolAdapter(search_tool)



async def main():
    
    model_client=OpenAIChatCompletionClient(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
        )
    agent = AssistantAgent(
        name="jarvis",
        tools=[tool],
        model_client=model_client,
        system_message="use search_tool for searching the web",
    )
    team=RoundRobinGroupChat(
        name="agent jarvis",
        participants=[agent],
        max_turns=2,

    )
    await Console(team.run_stream(task="what is current news about pm modi "))

if __name__ == "__main__":
    asyncio.run(main())
