import os
import asyncio
from TTS import TTS
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
load_dotenv()
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
llm = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
)
async def main(topic=""):
    # response= await llm.create([UserMessage(content=["in the autogen agentic framework what are the library present list the name"],source="User")])
    # print("response :",response.content)
    
    
    
    supporter=AssistantAgent(
        name="Hari",
        system_message=f"You are Hari , a supporter agent  in a debate for the topic: {topic}..You will be debating against Ram,which is a critic agent",
        model_client=llm
        
    )
    critic=AssistantAgent(
        name="Ram",
        system_message=f"You are Ram , a critic agent  in a debate for the topic: {topic}..You will be debating against Hari,which is a supporter agent",
        model_client=llm
        
    )
    team=RoundRobinGroupChat(
        participants=[supporter,critic],
        max_turns=4

    )
    response=await team.run(task="start the debate.")
    return response
    
    

    
    


if __name__=='__main__':
    response=asyncio.run(main(topic=" online education is good or bad. "))
    print("Debate Summary:")
    print("=="*80)

    for message in response.messages:
        print(f"Source: {message.source}")

        print(f"Content:\n{message.content}")
        TTS(message.content)
        print("--" * 80)


