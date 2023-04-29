from argparse import ArgumentParser
from sys import stdin

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

parser = ArgumentParser()
parser.add_argument("text", type=str, nargs="?", default=stdin)
args = parser.parse_args()

prompt = PromptTemplate(
    template="""
    You will be given a poker match outcome in {input_language}.
    Your response should be from the perspective of the player with the following username: {username}.
    Leave out any comments regarding bankroll or taking notes on opponents, assume the other players do not bluff.
    You can assume the other players could have either had a really bad hand or a good hand.
    First, you will give feedback on what {username} could have done better.
    Second, you will enumerate all possible hands that would have been better than what {username} had.
    Here is the poker match: {text}
    """,
    input_variables=["input_language", "username", "text"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

chat = ChatOpenAI(
    temperature=0.9,
    model_name="gpt-4"
)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

print(prompt.template)
print(
    chat(chat_prompt.format_prompt(input_language="Dutch", username="bitgo1141", text=args.text).to_messages()).content
)
