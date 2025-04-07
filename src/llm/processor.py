from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate

def process_data_with_llm(input_json_path, system_template):
    llm = ChatOpenAI(
        base_url="http://23.189.104.104:7777/v1",
        api_key="dummy_key",
        model="deepseek-v3",
        temperature=0.55
    )
    with open(input_json_path, "r", encoding="utf-8") as file:
        data = file.read()
    messages = [SystemMessagePromptTemplate.from_template(system_template)]
    prompt = ChatPromptTemplate.from_messages(messages)
    final_prompt = prompt.invoke({"data": data})
    response = llm.invoke(final_prompt)
    return response.content