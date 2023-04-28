from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(
    model_name="gpt-4-0314"
)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
chain.run("colorful socks")
