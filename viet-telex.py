from argparse import ArgumentParser
from sys import stdin

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

# https://stackoverflow.com/questions/49476326/displaying-unicode-in-powershell
# https://github.com/hwchase17/langchain/issues/1660#issuecomment-1469320129

parser = ArgumentParser()
parser.add_argument("text", type=str, nargs="?", default=stdin)
args = parser.parse_args()

prompt = PromptTemplate(
    template="""
    The user will send you input and you should return some output.
    
    Here comes an example of your input: a vietnamese word and an English word (a Vietnamese word is sometimes 2 words 
    separated by a space, the English word is always on the right): 
    
    | phẳng       | flat       |
    | công việc   | work       |
    | kinh nghiệm | experience |
    | cột         | column     |
    | nó          | it         |
    | nói         | said       |
    | xe          | car        |
    | giư         | hold       |
    | giai điệu   | melody     |
    
    The output should be a json object.
    The key should be a Vietnamese word.
    The value is another json object, it contains the following keys: telex, translation, ntd, tm.
    The telex key should contain a value which is the word that the user should type in order to get the Vietnamese word. 
    E.g. to type phố you should type pho-os: the extra o is for ô and the extra s is for ó, together they combine as ố.
    The translation key should contain a value which is the English translation:
    
    {{
      "hình dạng": {{
        "telex": "hinhf ddangj",
        "translation": "shape",
      }},
      "kinh nghiệm": {{
        "telex": "kinh nghiemej",
        "translation": "experience",
      }}
    }}
    
    Here is the table for the non-tonal diacritics (the first row is the header of the table):
    
    | Character | Keys pressed | Sample input     | Sample output |
    |-----------|--------------|------------------|---------------|
    | ă         | aw           | trang-w          | trăng         |
    | â         | aa           | can-a            | cân           |
    | đ         | dd           | d-dau-a          | đâu           |
    | ê         | ee           | d-dem-e          | đêm           |
    | ô         | oo           | nho-o            | nhô           |
    | ơ         | ow or [      | mo-w or m[       | mơ            |
    | ư         | uw or w or ] | tu-w or tw or t] | tư            |
    
    Here is the table for the tone markings (the first row is the header of the table):
    
    | Tone                       | Keys added to syllable | Sample input | Sample output |
    |----------------------------|------------------------|--------------|---------------|
    | Ngang (level)              | z or nothing           | ngang        | ngang         |
    | Huyền (falling)            | f                      | huyeenf      | huyền         |
    | Sắc (rising)               | s                      | sawcs        | sắc           |
    | Hỏi (dipping-rising)       | r                      | hoir         | hỏi           |
    | Ngã (rising glottalized)   | x                      | ngax         | ngã           |
    | Nặng (falling glottalized) | j                      | nawngj       | nặng          |
    | ư                          | uw                     | tuw          | tư            |
    
    Here comes the input:
        {text}
    """,
    input_variables=["text"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

chat = ChatOpenAI(
    temperature=0.9,
    model_name="gpt-4",
)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
final_text = chat_prompt.format_prompt(text=args.text)

print(final_text.to_string())
print('Number of tokens used: ', chat.get_num_tokens(final_text.to_string()))
print(
    chat(final_text.to_messages()).content)
