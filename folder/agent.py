import os
import openai
import sys

sys.path.append("../..")
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import time
import json
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
import tiktoken

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

from pydantic import BaseModel

persist_directory = "docs/chroma1/"
embedding = OpenAIEmbeddings()
mydb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
import pandas as pd

df = pd.read_csv("data.csv")

col = df.columns
mylist = []


for l in range(1, len(df)):
    stn = ""
    k = 1
    i = df.loc[l]
    # print("i : ",i)
    for j in i[1:]:
        # print("j : ",j)
        stn = stn + " , " + str(col[k]) + " is " + str(j)
        k = k + 1
    mylist.append(stn)
mydb = Chroma.from_texts(mylist, embedding=embedding)
history = ""


def vectordb(ques: str = ""):
    doc = mydb.max_marginal_relevance_search(ques, k=3)
    s = ""
    for i in doc:
        s = s + i.page_content
    return s


llm = OpenAI(temperature=0)

""" tools = load_tools(["serpapi"], llm=llm)

agent_executor = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
) """

conversation_history = []
recent_messages = []
n = 10  # number of messages to hold in conversation history

""" Introduction: Welcome the user and inquire about their specific product interest or mobile phone requirements, or the exact model they are looking for. """
""" Solution Presentation: Provide detailed information about the recommended products or mobile phones based on the user's preferences and needs. """
""" Feedback and Follow-Up: Gather feedback on the provided information and offer further assistance or follow-up options if needed. """
""" Courteous Farewell: Thank the user for using the platform and encourage them to return for any future inquiries or assistance. """

conversation_stages = [
    "Introduction: Greet the user and inquire about their specific product interest, mobile phone model they are interested in, or mobile phone requirements.",
    "Solution Presentation: Elaborate on the mobile phone's data fetched from the database. Provide advice about the product, pros and cons, compare it with other mobile phones, as fits the user's message/query.",
    "Courteous Farewell: Thank the user for using the platform and encourage them to return for any future inquiries or assistance.",
    "Out of scope message: The user is taking the conversation outside the 'mobile phone' domain, remind them that you are strictly a phone expert and can only converse about that. However, if they're making small talk, indulge them a little while reminding them of your actual prupose as a phone expert.",
]

conversation_stage = 1


function_descriptions = [
    {
        "name": "process_conversation_stage",
        "description": "Take in conversation stage and process it.",
        "parameters": {
            "type": "object",
            "properties": {
                "conversation_stage": {
                    "type": "number",
                    "description": "The stage the conversation is at.",
                    "example": 1,
                }
            },
            "required": ["conversation_stage"],
        },
    }
]

prompt_one = ""

prompt_two = ""


def add_message(message):
    global conversation_history
    global recent_messages
    global n

    # Append the new message to the primary conversation list
    conversation_history.append(message)

    # Update the recent messages list to contain the n most recent messages
    recent_messages.clear()  # Clear the current recent messages list

    # Use slicing to get the most recent messages from the conversation list
    recent_messages.extend(conversation_history[-n:])


def determine_stage(prompt_one, recent_messages):
    model_name = "gpt-3.5-turbo"

    encoding = tiktoken.encoding_for_model(model_name)

    token_count = len(encoding.encode(prompt_one))
    # print(prompt_one)
    # print(token_count)
    if token_count > 3200:
        model_name = "gpt-3.5-turbo-16k"

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_one}],
        temperature=0,
    )
    # print(completion)
    response_content = completion["choices"][0]["message"]["content"]
    result = response_content
    # print(result)
    return result


def run_conversation(prompt_two, recent_messages):
    global conversation_history
    global memory_track

    completion = ""

    model_name = "gpt-3.5-turbo"

    encoding = tiktoken.encoding_for_model(model_name)

    token_count = len(encoding.encode(prompt_two))

    # print(token_count)
    if token_count > 3200:
        model_name = "gpt-3.5-turbo-16k"

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_two}],
        temperature=0.9,
    )

    response_content = completion["choices"][0]["message"]["content"]
    result = response_content

    add_message("Mobile Phone Expert: " + result + "\n")

    return result


def process_input(user_input):
    global agent_executor
    global conversation_history
    global recent_messages
    global conversation_stages
    global conversation_stage

    db_count = 0
    data_from_vector_db = ""
    while True:
        user_input = input("User: ")
        add_message("User: " + user_input + "\n")

        if db_count < 3:
            db_count += 1
            data_from_vector_db = data_from_vector_db + "|" + vectordb(user_input)

        prompt_one = f"""Following is a list of conversation stages and a conversation history.
Use the conversation history to decide the next immediate stage of a 'mobile phone advisory conversation' which the mobile phone expert should proceed to.

Only answer with a number between '1' and '8' with a best guess of what stage should the conversation continue with.
Do not answer anything else nor add anything to your answer.

Conversation Stages:
1. Introduction: If this is the mobile phone experts's first message, then pick this stage.
2. Detailed Solution Presentation: If the conversation history has sufficient information about the user's mobile phone requirements and preferences, then pick this stage.
3. Closure and Courteous Farewell: If the conversation history shows that the user is ending the conversation, or that the conversation is taking an uncomfortable route, then pick this stage.
4. Out of scope question: If the latest user message is taking the conversation outside the 'mobile phone assistance' domain, then pick this stage.

Conversation History:
{"".join(recent_messages)}

"""
        conversation_stage = int(determine_stage(prompt_one, recent_messages))

        prompt_two = f"""You are a Mobile Phone Expert Customer Support Agent.
A user/customer has contacted you to talk about and seek advice about a mobile phone they are interested in.
Your end goal is to attempt to help them navigate their thoughts and decide the best mobile phone for them. You are provided with
specific data from a database about the phone the user is interested in. When making factual statements about that model, you will ONLY
use that data in your response. You will talk like a real person and avoid making lists.

You must respond according to the conversation history and the stage of the conversation you are at.
Only generate one response at a time! 

Data From Mobile Phone Database:
{data_from_vector_db}

Conversation history: 
{''.join(recent_messages)}

Current conversation stage: 
{conversation_stages[conversation_stage-1]}

Mobile Phone Expert:
"""
        # print(prompt_two)
        """ if conversation_stage == 2:
            search_result = agent_executor.invoke(
                {
                    "input": f"Conversation History: {'<END_OF_TURN>'.join(recent_messages)}"
                }
            )
            search_result = search_result["output"]
            search_result = (
                "Mobile Phone Expert: Thank you for your patience! I ran a quick search and here's what I found: "
                + search_result
                + "\n"
            )
            add_message(search_result)
            print(search_result) """

        result = run_conversation(prompt_two, recent_messages)
        text = text + result
        print("Mobile Phone Expert: " + result)

    return text


process_input()
