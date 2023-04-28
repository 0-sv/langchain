from argparse import ArgumentParser
from sys import stdin

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

parser = ArgumentParser()
parser.add_argument("text", type=str, nargs="?", default=stdin)
args = parser.parse_args()

chat = ChatOpenAI(temperature=0.9)
print(chat([HumanMessage(content=args.text)]))
