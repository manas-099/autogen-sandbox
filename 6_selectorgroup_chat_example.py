# “Team Brainstorm for a Product Launch”
import autogen_agentchat as ag
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
import os
import asyncio

load_dotenv()
gemini_api_key=os.getenv('GOOGLE_API_KEY')
print("Gemini key loaded:", gemini_api_key is not None)
async def main() -> None:
    
    model_client=OpenAIChatCompletionClient(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
        )

    idea_agent = AssistantAgent(
        name="IdeaAgent",
        description="Generates creative product ideas for new tech gadgets.",
        system_message="You are an innovative agent. Brainstorm one original product idea.",
        model_client=model_client,
    )

    marketing_agent = AssistantAgent(
        name="MarketingAgent",
        description="Creates catchy taglines and marketing angles for product ideas.",
        system_message="You are a marketing expert. Provide a tagline and positioning statement for the product.",
        model_client=model_client,
    )

    review_agent = AssistantAgent(
        name="ReviewAgent",
        description="Critically reviews the idea and marketing and recommends improvements.",
        system_message="You are a thoughtful reviewer. Evaluate the product idea and marketing, then suggest one improvement.",
        model_client=model_client,
    )

    # --- Termination condition: stop after 6 messages ---
    termination = MaxMessageTermination(max_messages=6)

    # --- Custom selector prompt message ---
    selector_prompt = """
    Select the next agent to speak based on these roles:
    {roles}

    Current conversation:
    {history}

    Pick one agent to contribute next.
    """

    # --- Build the SelectorGroupChat team ---
    team = SelectorGroupChat(
        participants=[idea_agent, marketing_agent, review_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False  # prevent same agent from speaking twice in a row
    )
    await Console(team.run_stream(task="Let's brainstorm a new eco-friendly smart home gadget."))


# task = ""

# # --- Run the team and print conversation log ---
# result = await team.run(task=task)
# for turn in result.history:
#     print(f"{turn.agent_name}: {turn.message.content}")
if __name__=='__main__':
    asyncio.run(main())
