import traceback
import boto3
import os
import json
import re
import uuid
import time
import base64
import info
import PyPDF2
import csv
import utils
import asyncio
import streamlit
import pandas as pd

from io import BytesIO
from PIL import Image
from langchain_aws import ChatBedrock
from botocore.config import Config
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import tool
from langchain.docstore.document import Document
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from urllib import parse
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from multiprocessing import Process, Pipe
from streamlit.delta_generator import DeltaGenerator
from langchain_core.callbacks.base import BaseCallbackHandler

logger = utils.CreateLogger("chat")

userId = uuid.uuid4().hex
map_chain = dict()

checkpointers = dict()
memorystores = dict()

checkpointer = MemorySaver()
memorystore = InMemoryStore()

checkpointers[userId] = checkpointer
memorystores[userId] = memorystore

reasoning_mode = 'Disable'

def initiate():
    global userId
    global memory_chain, checkpointers, memorystores, checkpointer, memorystore

    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

    if userId in map_chain:
            # print('memory exist. reuse it!')
            memory_chain = map_chain[userId]

            checkpointer = checkpointers[userId]
            memorystore = memorystores[userId]
    else:
        # print('memory does not exist. create new one!')
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

        checkpointer = MemorySaver()
        memorystore = InMemoryStore()

        checkpointers[userId] = checkpointer
        memorystores[userId] = memorystore

initiate()

config = utils.load_config()

bedrock_region = config["region"] if "region" in config else "us-west-2"

projectName = config["projectName"] if "projectName" in config else "mcp-rag"

accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")

s3_prefix = 'docs'
s3_image_prefix = 'images'

knowledge_base_role = config["knowledge_base_role"] if "knowledge_base_role" in config else None
if knowledge_base_role is None:
    raise Exception ("No Knowledge Base Role")

collectionArn = config["collectionArn"] if "collectionArn" in config else None
if collectionArn is None:
    raise Exception ("No collectionArn")

vectorIndexName = projectName

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

knowledge_base_name = projectName
numberOfDocs = 4

MSG_LENGTH = 100

doc_prefix = s3_prefix+'/'

model_name = "Claude 3.5 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
number_of_models = len(models)
model_id = models[0]["model_id"]
debug_mode = "Enable"
multi_region = "Disable"

client = boto3.client(
    service_name='bedrock-agent',
    region_name=bedrock_region
)

mcp_json = ""
def update(modelName, debugMode, multiRegion, mcp):
    global model_name, model_id, model_type, debug_mode, multi_region
    global models, mcp_json

    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")

        models = info.get_model_info(model_name)
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]

    if debug_mode != debugMode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")

    if multi_region != multiRegion:
        multi_region = multiRegion
        logger.info(f"multi_region: {multi_region}")

    mcp_json = mcp
    logger.info(f"mcp_json: {mcp_json}")

def clear_chat_history():
    memory_chain = []
    map_chain[userId] = memory_chain

def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])
    else:
        memory_chain.chat_memory.add_ai_message(msg)

selected_chat = 0
def get_chat(extended_thinking):
    global selected_chat, model_type

    logger.info(f"models: {models}")
    logger.info(f"selected_chat: {selected_chat}")

    profile = models[selected_chat]
    # print('profile: ', profile)

    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    number_of_models = len(models)

    logger.info(f"LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}")

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:"

    # bedrock
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    if extended_thinking=='Enable':
        maxReasoningOutputTokens=64000
        logger.info(f"extended_thinking: {extended_thinking}")
        thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

        parameters = {
            "max_tokens":maxReasoningOutputTokens,
            "temperature":1,
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "stop_sequences": [STOP_SEQUENCE]
        }
    else:
        parameters = {
            "max_tokens":maxOutputTokens,
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [STOP_SEQUENCE]
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock,
        model_kwargs=parameters,
        region_name=bedrock_region
    )

    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content

    logger.info(f"{i}: {text}, metadata:{doc.metadata}")

def translate_text(text):
    chat = get_chat(extended_thinking="Disable")

    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags. Put it in <result> tags."
    )
    human = "<article>{text}</article>"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)

    if isKorean(text)==False :
        input_language = "English"
        output_language = "Korean"
    else:
        input_language = "Korean"
        output_language = "English"

    chain = prompt | chat
    try:
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        msg = result.content
        logger.info(f"translated text: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def check_grammer(text):
    chat = get_chat(extended_thinking="Disable")

    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."
        )
    else:
        system = (
            "Here is pieces of article, contained in <article> tags. Find the error in the sentence and explain it, and add the corrected sentence at the end of your answer."
        )

    human = "<article>{text}</article>"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)

    chain = prompt | chat
    try:
        result = chain.invoke(
            {
                "text": text
            }
        )

        msg = result.content
        logger.info(f"result of grammer correction: {msg}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM")

    return msg

reference_docs = []
# api key to get weather information in agent
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)

# api key for weather
weather_api_key = ""
try:
    pass
    # get_weather_api_secret = secretsmanager.get_secret_value(
    #     SecretId=f"openweathermap-{projectName}"
    # )
    # #print('get_weather_api_secret: ', get_weather_api_secret)
    # secret = json.loads(get_weather_api_secret['SecretString'])
    # #print('secret: ', secret)
    # weather_api_key = secret['weather_api_key']

except Exception as e:
    raise e

# api key to use LangSmith
langsmith_api_key = ""
try:
    pass
    # get_langsmith_api_secret = secretsmanager.get_secret_value(
    #     SecretId=f"langsmithapikey-{projectName}"
    # )
    # #print('get_langsmith_api_secret: ', get_langsmith_api_secret)
    # secret = json.loads(get_langsmith_api_secret['SecretString'])
    # #print('secret: ', secret)
    # langsmith_api_key = secret['langsmith_api_key']
    # langchain_project = secret['langchain_project']
except Exception as e:
    raise e

if langsmith_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# secret of code interpreter
code_interpreter_api_key = ""
try:
    pass
    # get_code_interpreter_api_secret = secretsmanager.get_secret_value(
    #     SecretId=f"code-interpreter-{projectName}"
    # )
    # #print('get_code_interpreter_api_secret: ', get_code_interpreter_api_secret)
    # secret = json.loads(get_code_interpreter_api_secret['SecretString'])
    # #print('secret: ', secret)
    # code_interpreter_api_key = secret['code_interpreter_api_key']
    # code_interpreter_project = secret['project_name']
    # code_interpreter_id = secret['code_interpreter_id']

    # logger.info(f"code_interpreter_id: {code_interpreter_id}")
except Exception as e:
    raise e

if code_interpreter_api_key:
    os.environ["RIZA_API_KEY"] = code_interpreter_api_key

# api key to use Tavily Search
tavily_key = tavily_api_wrapper = ""
try:
    pass
    # get_tavily_api_secret = secretsmanager.get_secret_value(
    #     SecretId=f"tavilyapikey-{projectName}"
    # )
    # #print('get_tavily_api_secret: ', get_tavily_api_secret)
    # secret = json.loads(get_tavily_api_secret['SecretString'])
    # #print('secret: ', secret)

    # if "tavily_api_key" in secret:
    #     tavily_key = secret['tavily_api_key']
    #     #print('tavily_api_key: ', tavily_api_key)

    #     if tavily_key:
    #         tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
    #         #     os.environ["TAVILY_API_KEY"] = tavily_key

    #     else:
    #         logger.info(f"tavily_key is required.")
except Exception as e:
    logger.info(f"Tavily credential is required: {e}")
    raise e

# api key to use Tavily Search
firecrawl_key = ""
try:
    pass
    # get_firecrawl_secret = secretsmanager.get_secret_value(
    #     SecretId=f"firecrawlapikey-{projectName}"
    # )
    # secret = json.loads(get_firecrawl_secret['SecretString'])

    # if "firecrawl_api_key" in secret:
    #     firecrawl_key = secret['firecrawl_api_key']
    #     # print('firecrawl_api_key: ', firecrawl_key)
except Exception as e:
    logger.info(f"Firecrawl credential is required: {e}")
    raise e

def get_references(docs):
    reference = ""
    for i, doc in enumerate(docs):
        page = ""
        if "page" in doc.metadata:
            page = doc.metadata['page']
            #print('page: ', page)
        url = ""
        if "url" in doc.metadata:
            url = doc.metadata['url']
            logger.info(f"url: {url}")
        name = ""
        if "name" in doc.metadata:
            name = doc.metadata['name']
            #print('name: ', name)

        sourceType = ""
        if "from" in doc.metadata:
            sourceType = doc.metadata['from']
        else:
            # if useEnhancedSearch:
            #     sourceType = "OpenSearch"
            # else:
            #     sourceType = "WWW"
            sourceType = "WWW"

        #print('sourceType: ', sourceType)

        #if len(doc.page_content)>=1000:
        #    excerpt = ""+doc.page_content[:1000]
        #else:
        #    excerpt = ""+doc.page_content
        excerpt = ""+doc.page_content
        # print('excerpt: ', excerpt)

        # for some of unusual case
        #excerpt = excerpt.replace('"', '')
        #excerpt = ''.join(c for c in excerpt if c not in '"')
        excerpt = re.sub('"', '', excerpt)
        excerpt = re.sub('#', '', excerpt)
        excerpt = re.sub('\n', '', excerpt)
        logger.info(f"excerpt(quotation removed): {excerpt}")

        if page:
            reference += f"{i+1}. {page}page in [{name}]({url})), {excerpt[:30]}...\n"
        else:
            reference += f"{i+1}. [{name}]({url}), {excerpt[:30]}...\n"

    if reference:
        reference = "\n\n#### 관련 문서\n"+reference

    return reference

def tavily_search(query, k):
    docs = []
    try:
        tavily_client = TavilyClient(api_key=tavily_key)
        response = tavily_client.search(query, max_results=k)
        # print('tavily response: ', response)

        for r in response["results"]:
            name = r.get("title")
            if name is None:
                name = 'WWW'

            docs.append(
                Document(
                    page_content=r.get("content"),
                    metadata={
                        'name': name,
                        'url': r.get("url"),
                        'from': 'tavily'
                    },
                )
            )
    except Exception as e:
        logger.info(f"Exception: {e}")

    return docs

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags."
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)

    chain = prompt | chat
    try:
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )

        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def extract_thinking_tag(response, st):
    if response.find('<thinking>') != -1:
        status = response[response.find('<thinking>')+11:response.find('</thinking>')]
        logger.info(f"gent_thinking: {status}")

        if debug_mode=="Enable":
            st.info(status)

        if response.find('<thinking>') == 0:
            msg = response[response.find('</thinking>')+13:]
        else:
            msg = response[:response.find('<thinking>')]
        logger.info(f"msg: {msg}")
    else:
        msg = response

    return msg

def get_parallel_processing_chat(models, selected):
    global model_type
    profile = models[selected]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    maxOutputTokens = 4096
    logger.info(f'selected_chat: {selected}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:"

    # bedrock
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock,
        model_kwargs=parameters,
    )
    return chat

def print_doc(i, doc):
    if len(doc.page_content)>=100:
        text = doc.page_content[:100]
    else:
        text = doc.page_content

    logger.info(f"{i}: {text}, metadata:{doc.metadata}")

def grade_document_based_on_relevance(conn, question, doc, models, selected):
    chat = get_parallel_processing_chat(models, selected)
    retrieval_grader = get_retrieval_grader(chat)
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    # print(f"score: {score}")

    grade = score.binary_score
    if grade == 'yes':
        logger.info(f"---GRADE: DOCUMENT RELEVANT---")
        conn.send(doc)
    else:  # no
        logger.info(f"--GRADE: DOCUMENT NOT RELEVANT---")
        conn.send(None)

    conn.close()

def grade_documents_using_parallel_processing(question, documents):
    global selected_chat

    filtered_docs = []

    processes = []
    parent_connections = []

    for i, doc in enumerate(documents):
        #print(f"grading doc[{i}]: {doc.page_content}")
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)

        process = Process(target=grade_document_based_on_relevance, args=(child_conn, question, doc, models, selected_chat))
        processes.append(process)

        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    for process in processes:
        process.start()

    for parent_conn in parent_connections:
        relevant_doc = parent_conn.recv()

        if relevant_doc is not None:
            filtered_docs.append(relevant_doc)

    for process in processes:
        process.join()

    return filtered_docs

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def get_retrieval_grader(chat):
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    # from langchain_core.output_parsers import PydanticOutputParser  # not supported for Nova
    # parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    # retrieval_grader = grade_prompt | chat | parser

    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def show_extended_thinking(st, result):
    # logger.info(f"result: {result}")
    if "thinking" in result.response_metadata:
        if "text" in result.response_metadata["thinking"]:
            thinking = result.response_metadata["thinking"]["text"]
            st.info(thinking)

def grade_documents(question, documents):
    logger.info(f"###### grade_documents ######")

    logger.info(f"start grading...")

    filtered_docs = []
    if multi_region == 'Enable':  # parallel processing
        filtered_docs = grade_documents_using_parallel_processing(question, documents)

    else:
        # Score each doc
        llm = get_chat(extended_thinking="Disable")
        retrieval_grader = get_retrieval_grader(llm)
        for i, doc in enumerate(documents):
            # print('doc: ', doc)
            print_doc(i, doc)

            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            # print("score: ", score)

            grade = score.binary_score
            # print("grade: ", grade)
            # Document relevant
            if grade.lower() == "yes":
                logger.info(f"---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            # Document not relevant
            else:
                logger.info(f"---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                continue

    return filtered_docs

####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    llm = get_chat(extended_thinking="Disable")

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )

    human = "Question: {input}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", human)
    ])

    history = memory_chain.load_memory_variables({})["chat_history"]

    chain = prompt | llm | StrOutputParser()
    try:
        stream = chain.stream(
            {
                "history": history,
                "input": query,
            }
        )
        logger.info(f"stream: {stream}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM: "+err_msg)

    return stream

def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )
        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"

        content_type = utils.get_contents_type(file_name)
        logger.info(f"content_type: {content_type}")

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{file_name}"
        else:
            s3_key = f"{s3_prefix}/{file_name}"

        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }

        response = s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes
        )
        logger.info(f"upload response: {response}")

        #url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        url = path+'/'+s3_image_prefix+'/'+parse.quote(file_name)
        return url

    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

def extract_and_display_s3_images(text, s3_client):
    """
    Extract S3 URLs from text, download images, and return them for display
    """
    s3_pattern = r"https://[\w\-\.]+\.s3\.amazonaws\.com/[\w\-\./]+"
    s3_urls = re.findall(s3_pattern, text)

    images = []
    for url in s3_urls:
        try:
            bucket = url.split(".s3.amazonaws.com/")[0].split("//")[1]
            key = url.split(".s3.amazonaws.com/")[1]

            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()

            image = Image.open(BytesIO(image_data))
            images.append(image)

        except Exception as e:
            err_msg = f"Error downloading image from S3: {str(e)}"
            logger.info(f"{err_msg}")
            continue

    return images

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    logger.info(f"prelinspare: {len(lines)}")

    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]
    #columns_to_metadata = ["type","Source"]
    logger.info(f"columns: {columns}")

    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    logger.info(f"docs[0]: {docs[0]}")

    return docs

def get_summary(docs):
    llm = get_chat(extended_thinking="Disable")

    text = ""
    for doc in docs:
        text = text + doc

    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else:
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )

    human = "<article>{text}</article>"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)

    chain = prompt | llm
    try:
        result = chain.invoke(
            {
                "text": text
            }
        )

        summary = result.content
        logger.info(f"esult of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM")

    return summary

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    logger.info(f"s3_bucket: {s3_bucket}, s3_prefix: {s3_prefix}, s3_file_name: {s3_file_name}")

    contents = ""
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))

        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)

    elif file_type == 'txt' or file_type == 'md':
        contents = doc.get()['Body'].read().decode('utf-8')

    logger.info(f"contents: {contents}")
    new_contents = str(contents).replace("\n"," ")
    logger.info(f"length: {len(new_contents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    )
    texts = text_splitter.split_text(new_contents)
    if texts:
        logger.info(f"exts[0]: {texts[0]}")

    return texts

def summary_of_code(code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )

    human = "<article>{code}</article>"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)

    llm = get_chat(extended_thinking="Disable")

    chain = prompt | llm
    try:
        result = chain.invoke(
            {
                "code": code
            }
        )

        summary = result.content
        logger.info(f"result of code summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM")

    return summary

def summary_image(img_base64, instruction):
    llm = get_chat(extended_thinking="Disable")

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요."

    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]

    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try:
            result = llm.invoke(messages)

            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            raise Exception ("Not able to request to LLM")

    return extracted_text

def extract_text(img_base64):
    multimodal = get_chat(extended_thinking="Disable")
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."

    extracted_text = ""
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]

    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try:
            result = multimodal.invoke(messages)

            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            # raise Exception ("Not able to request to LLM")

    logger.info(f"xtracted_text: {extracted_text}")
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)
def get_summary_of_uploaded_file(file_name, st):
    file_type = file_name[file_name.rfind('.')+1:len(file_name)]
    logger.info(f"file_type: {file_type}")

    if file_type == 'csv':
        docs = load_csv_document(file_name)
        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        logger.info(f"contexts: {contexts}")

        msg = get_summary(contexts)

    elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
        texts = load_document(file_type, file_name)

        if len(texts):
            docs = []
            for i in range(len(texts)):
                docs.append(
                    Document(
                        page_content=texts[i],
                        metadata={
                            'name': file_name,
                            # 'page':i+1,
                            'url': path+'/'+doc_prefix+parse.quote(file_name)
                        }
                    )
                )
            logger.info(f"docs[0]: {docs[0]}")
            logger.info(f"docs size: {len(docs)}")

            contexts = []
            for doc in docs:
                contexts.append(doc.page_content)
            logger.info(f"contexts: {contexts}")

            msg = get_summary(contexts)
        else:
            msg = "문서 로딩에 실패하였습니다."

    elif file_type == 'py' or file_type == 'js':
        s3r = boto3.resource("s3")
        doc = s3r.Object(s3_bucket, s3_prefix+'/'+file_name)

        contents = doc.get()['Body'].read().decode('utf-8')

        #contents = load_code(file_type, object)

        msg = summary_of_code(contents, file_type)

    elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
        logger.info(f"multimodal: {file_name}")

        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )
        if debug_mode=="Enable":
            status = "이미지를 가져옵니다."
            logger.info(f"status: {status}")
            st.info(status)

        image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+file_name)
        # print('image_obj: ', image_obj)

        image_content = image_obj['Body'].read()
        img = Image.open(BytesIO(image_content))

        width, height = img.size
        logger.info(f"width: {width}, height: {height}, size: {width*height}")

        isResized = False
        while(width*height > 5242880):
            width = int(width/2)
            height = int(height/2)
            isResized = True
            logger.info(f"width: {width}, height: {height}, size: {width*height}")

        if isResized:
            img = img.resize((width, height))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # extract text from the image
        if debug_mode=="Enable":
            status = "이미지에서 텍스트를 추출합니다."
            logger.info(f"status: {status}")
            st.info(status)

        text = extract_text(img_base64)
        # print('extracted text: ', text)

        if text.find('<result>') != -1:
            extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
            # print('extracted_text: ', extracted_text)
        else:
            extracted_text = text

        if debug_mode=="Enable":
            logger.info(f"### 추출된 텍스트\n\n{extracted_text}")
            print('status: ', status)
            st.info(status)

        if debug_mode=="Enable":
            status = "이미지의 내용을 분석합니다."
            logger.info(f"status: {status}")
            st.info(status)

        image_summary = summary_image(img_base64, "")
        logger.info(f"image summary: {image_summary}")

        if len(extracted_text) > 10:
            contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
        else:
            contents = f"## 이미지 분석\n\n{image_summary}"
        logger.info(f"image content: {contents}")

        msg = contents

    global fileId
    fileId = uuid.uuid4().hex
    # print('fileId: ', fileId)

    return msg

####################### LangChain #######################
# Image Summarization
#########################################################
def get_image_summarization(object_name, prompt, st):
    # load image
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )

    if debug_mode=="Enable":
        status = "이미지를 가져옵니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_image_prefix+'/'+object_name)
    # print('image_obj: ', image_obj)

    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))

    width, height = img.size
    logger.info(f"width: {width}, height: {height}, size: {width*height}")

    isResized = False
    while(width*height > 5242880):
        width = int(width/2)
        height = int(height/2)
        isResized = True
        logger.info(f"width: {width}, height: {height}, size: {width*height}")

    if isResized:
        img = img.resize((width, height))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # extract text from the image
    if debug_mode=="Enable":
        status = "이미지에서 텍스트를 추출합니다."
        logger.info(f"status: {status}")
        st.info(status)

    text = extract_text(img_base64)
    logger.info(f"extracted text: {text}")

    if text.find('<result>') != -1:
        extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
        # print('extracted_text: ', extracted_text)
    else:
        extracted_text = text

    if debug_mode=="Enable":
        status = f"### 추출된 텍스트\n\n{extracted_text}"
        logger.info(f"status: {status}")
        st.info(status)

    if debug_mode=="Enable":
        status = "이미지의 내용을 분석합니다."
        logger.info(f"status: {status}")
        st.info(status)

    image_summary = summary_image(img_base64, prompt)

    if text.find('<result>') != -1:
        image_summary = image_summary[image_summary.find('<result>')+8:image_summary.find('</result>')]
    logger.info(f"image summary: {image_summary}")

    if len(extracted_text) > 10:
        contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
    else:
        contents = f"## 이미지 분석\n\n{image_summary}"
    logger.info(f"image contents: {contents}")

    return contents


####################### Bedrock Agent #######################
# RAG using Lambda
#############################################################
def get_rag_prompt(text):
    # print("###### get_rag_prompt ######")
    llm = get_chat(extended_thinking="Disable")
    # print('model_type: ', model_type)

    if model_type == "nova":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
            )
        else:
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
            )

        human = (
            "Question: {question}"

            "Reference texts: "
            "{context}"
        )

    elif model_type == "claude":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else:
            system = (
                "You will be acting as a thoughtful advisor."
                "Here is pieces of context, contained in <context> tags."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
                "Put it in <result> tags."
            )

        human = (
            "<question>"
            "{question}"
            "</question>"

            "<context>"
            "{context}"
            "</context>"
        )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)

    rag_chain = prompt | llm

    return rag_chain

def retrieve_knowledge_base(query):
    lambda_client = boto3.client(
        service_name='lambda',
        region_name=bedrock_region
    )

    functionName = f"lambda-rag-for-{projectName}"
    logger.info(f"functionName: {functionName}")

    try:
        payload = {
            'function': 'search_rag',
            'knowledge_base_name': knowledge_base_name,
            'keyword': query,
            'top_k': numberOfDocs,
            'grading': "Enable",
            'model_name': model_name,
            'multi_region': multi_region
        }
        logger.info(f"payload: {payload}")

        output = lambda_client.invoke(
            FunctionName=functionName,
            Payload=json.dumps(payload),
        )
        payload = json.load(output['Payload'])
        logger.info(f"response: {payload['response']}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    return payload['response']

def run_rag_with_knowledge_base(query, st):
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {query}")

    relevant_context = retrieve_knowledge_base(query)
    logger.info(f"relevant_context: {relevant_context}")
    st.info(f"RAG 검색을 완료했습니다.")
    st.info(f"{relevant_context}")

    rag_chain = get_rag_prompt(query)

    msg = ""
    try:
        result = rag_chain.invoke(
            {
                "question": query,
                "context": relevant_context
            }
        )
        logger.info(f"result: {result}")

        msg = result.content
        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]

    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")
        raise Exception ("Not able to request to LLM")

    return msg, reference_docs


def streamlit_callback(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.
    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """
    return parent_container


####################### Bedrock Agent #######################
# Bedrock Agent (Multi agent collaboration)
#############################################################

def create_agent(tools, historyMode):
    tool_node = ToolNode(tools)

    chatModel = get_chat(extended_thinking="Disable")
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State, config):
        logger.info(f"###### call_model ######")
        logger.info(f"state: {state['messages']}")

        last_message = state['messages'][-1]
        logger.info(f"last message: {last_message}")

        if isKorean(state["messages"][0].content)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "한국어로 답변하세요."
            )
        else:
            system = (
                "You are a conversational AI designed to answer in a friendly way to a question."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            chain = prompt | model

            response = chain.invoke(state["messages"], )
            # logger.info(f"call_model response: {response}")
            logger.info(f"call_model: {response.content}")

        except Exception:
            response = AIMessage(content="답변을 찾지 못하였습니다.")

            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            # raise Exception ("Not able to request to LLM")

        return {"messages": [response]}

    def should_continue(state: State) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")

        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_name = last_message.tool_calls[-1]['name']
            logger.info(f"--- CONTINUE: {tool_name} ---")
            return "continue"
        else:
            logger.info(f"--- END ---")
            return "end"

    def draw_response(state: State):
        logger.info(f"###### draw_response ######")

        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_name = last_message.tool_calls[-1]['name']
            logger.info(f"--- CONTINUE: {tool_name} ---")
            return "continue"
        else:
            logger.info(f"--- END ---")
            return "end"

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile()

    def buildChatAgentWithHistory():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_node("draw_response", draw_response)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        # workflow.add_edge("agent", draw_response)
        # workflow.add_edge("draw_response", "agent")
        workflow.add_edge("action", "agent")

        return workflow.compile(
            checkpointer=checkpointer,
            store=memorystore
        )

    # workflow
    if historyMode == "Enable":
        app = buildChatAgentWithHistory()
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": userId}
        }
    else:
        app = buildChatAgent()
        config = {
            "recursion_limit": 50
        }

    return app, config

# server_params = StdioServerParameters(
#   command="python",
#   args=["application/mcp-server.py"],
# )

def load_mcp_server_parameters():
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")

    command = ""
    args = []
    if mcpServers is not None:
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

            break

    return StdioServerParameters(
        command=command,
        args=args,
        env=env
    )

def load_multiple_mcp_server_parameters():
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")

    server_info = {}
    if mcpServers is not None:
        command = ""
        args = []
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

                server_info[server] = {
                    "command": command,
                    "args": args,
                    "env": env,
                    "transport": "stdio"
                }
            elif "url" in config:
                url = config["url"]
                server_info[server] = {
                    "url": url,
                    "transport": "sse",
                    # "reconnect": {
                    #     "enabled": True,
                    #     "maxAttempts": 5,
                    #     "delayMs": 2000
                    # }
                }
            else:
                server_info[server] = {
                    "command": command,
                    "args": args,
                    "transport": "stdio"
                }


    logger.info(f"server_info: {server_info}")

    return server_info

def tool_info(tools, st):
    tool_info = ""
    tool_list = []
    st.info("Tool 정보를 가져옵니다.")
    for tool in tools:
        tool_info += f"name: {tool.name}\n"
        if hasattr(tool, 'description'):
            tool_info += f"description: {tool.description}\n"
        tool_info += f"args_schema: {tool.args_schema}\n\n"
        tool_list.append(tool.name)
    # st.info(f"{tool_info}")
    st.info(f"Tools: {tool_list}")

def show_status_message(response, st):
    image_url = []
    references = []
    for i, re in enumerate(response):
        logger.info(f"message[{i}]: {re}")

        if i==len(response)-1:
            break

        if isinstance(re, AIMessage):
            logger.info(f"AIMessage: {re}")
            if re.content:
                logger.info(f"content: {re.content}")
                content = re.content
                if len(content) > 500:
                    content = content[:500] + "..."
                if debug_mode == "Enable":
                    st.info(f"{content}")
            if hasattr(re, 'tool_calls') and re.tool_calls:
                logger.info(f"Tool name: {re.tool_calls[0]['name']}")

                if 'args' in re.tool_calls[0]:
                    logger.info(f"Tool args: {re.tool_calls[0]['args']}")

                    args = re.tool_calls[0]['args']
                    if 'code' in args:
                        logger.info(f"code: {args['code']}")
                        if debug_mode == "Enable":
                            st.code(args['code'])
                    elif re.tool_calls[0]['args']:
                        if debug_mode == "Enable":
                            st.info(f"Tool name: {re.tool_calls[0]['name']}  \nTool args: {re.tool_calls[0]['args']}")
            # else:
            #     st.info(f"Tool name: {re.tool_calls[0]['name']}")

        elif isinstance(re, ToolMessage):
            if re.name:
                logger.info(f"Tool name: {re.name}")

                if re.content:
                    content = re.content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    logger.info(f"Tool result: {content}")

                    if debug_mode == "Enable":
                        st.info(f"Tool name: {re.name}  \nTool result: {content}")
                else:
                    if debug_mode == "Enable":
                        st.info(f"Tool name: {re.name}")
            try:
                # tavily
                if isinstance(re.content, str) and "Title:" in re.content and "URL:" in re.content and "Content:" in re.content:
                    logger.info("Tavily parsing...")
                    items = re.content.split("\n\n")
                    for i, item in enumerate(items):
                        logger.info(f"item[{i}]: {item}")
                        if "Title:" in item and "URL:" in item and "Content:" in item:
                            try:
                                # 정규식 대신 문자열 분할 방법 사용
                                title_part = item.split("Title:")[1].split("URL:")[0].strip()
                                url_part = item.split("URL:")[1].split("Content:")[0].strip()
                                content_part = item.split("Content:")[1].strip()

                                logger.info(f"title_part: {title_part}")
                                logger.info(f"url_part: {url_part}")
                                logger.info(f"content_part: {content_part}")

                                references.append({
                                    "url": url_part,
                                    "title": title_part,
                                    "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                                })
                            except Exception as e:
                                logger.info(f"파싱 오류: {str(e)}")
                                continue

                # check json format
                if isinstance(re.content, str) and (re.content.strip().startswith('{') or re.content.strip().startswith('[')):
                    tool_result = json.loads(re.content)
                    logger.info(f"tool_result: {tool_result}")
                else:
                    tool_result = re.content
                    logger.info(f"tool_result (not JSON): {tool_result}")
                print("tool result tupe", type(tool_result))

                if isinstance(tool_result, pd.DataFrame):
                    logger.info("Parse dataframe...............")
                    st.dataframe(tool_result)
                if "path" in tool_result:
                    logger.info(f"Path: {tool_result['path']}")

                    path = tool_result['path']
                    if isinstance(path, list):
                        for p in path:
                            logger.info(f"image: {p}")
                            if p.startswith('http') or p.startswith('https'):
                                st.image(p)
                                image_url.append(p)
                            else:
                                with open(p, 'rb') as f:
                                    image_data = f.read()
                                    st.image(image_data)
                                    image_url.append(p)
                    else:
                        logger.info(f"image: {path}")
                        try:
                            if path.startswith('http') or path.startswith('https'):
                                st.image(path)
                                image_url.append(path)
                            else:
                                with open(path, 'rb') as f:
                                    image_data = f.read()
                                    st.image(image_data)
                                    image_url.append(path)
                        except Exception as e:
                            logger.error(f"이미지 표시 오류: {str(e)}")
                            st.error(f"이미지를 표시할 수 없습니다: {str(e)}")

                # ArXiv
                if "papers" in tool_result:
                    logger.info(f"size of papers: {len(tool_result['papers'])}")

                    papers = tool_result['papers']
                    for paper in papers:
                        url = paper['url']
                        title = paper['title']
                        content = paper['abstract'][:100]
                        logger.info(f"url: {url}, title: {title}, content: {content}")

                        references.append({
                            "url": url,
                            "title": title,
                            "content": content
                        })

                if isinstance(tool_result, list):
                    logger.info(f"size of tool_result: {len(tool_result)}")
                    for i, item in enumerate(tool_result):
                        logger.info(f'item[{i}]: {item}')

                        # RAG
                        if "reference" in item:
                            logger.info(f"reference: {item['reference']}")

                            infos = item['reference']
                            url = infos['url']
                            title = infos['title']
                            source = infos['from']
                            logger.info(f"url: {url}, title: {title}, source: {source}")

                            references.append({
                                "url": url,
                                "title": title,
                                "content": item['contents'][:100]
                            })

                        # Others
                        if isinstance(item, str):
                            try:
                                item = json.loads(item)

                                # AWS Document
                                if "rank_order" in item:
                                    references.append({
                                        "url": item['url'],
                                        "title": item['title'],
                                        "content": item['context'][:100]
                                    })
                            except json.JSONDecodeError:
                                logger.info(f"JSON parsing error: {item}")
                                continue

            except:
                logger.info(f"fail to parsing..")
                pass
    return image_url, references

async def mcp_rag_agent_multiple(query, historyMode, st: streamlit):
    server_params = load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    async with  MultiServerMCPClient(server_params) as client:
        references = []
        ref = ""
        with st.status("thinking...", expanded=True, state="running") as status:
            tools = client.get_tools()
            logger.info(f"tools: {tools}")

            if debug_mode == "Enable":
                tool_info(tools, st)

            # langgraph agent
            agent, config = create_agent(tools, historyMode)

            response = await agent.ainvoke({"messages": query}, config)
            logger.info(f"response: {response}")

            result = response["messages"][-1].content
            logger.info(f"result: {result}")

            image_url, references = show_status_message(response["messages"], st)

            if references:
                ref = "\n\n### Reference\n"
            for i, reference in enumerate(references):
                ref += f"{i+1}. [{reference['title']}]({reference['url']}), {reference['content']}...\n"

            result += ref

        st.markdown(result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result,
            "images": image_url if image_url else []
        })

    return result

async def mcp_rag_agent_single(query, historyMode, st):
    server_params = load_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    async with stdio_client(server_params) as (read, write):
        # Open an MCP session to interact with the math_server.py tool.
        async with ClientSession(read, write) as session:
            # Initialize the session.
            await session.initialize()

            logger.info(f"query: {query}")

            # Load tools
            tools = await load_mcp_tools(session)
            logger.info(f"tools: {tools}")

            with st.status("thinking...", expanded=True, state="running") as status:
                if debug_mode == "Enable":
                    tool_info(tools, st)

                agent = create_agent(tools, historyMode)

                # Run the agent.
                agent_response = await agent.ainvoke({"messages": query})
                logger.info(f"agent_response: {agent_response}")

                if debug_mode == "Enable":
                    for i, re in enumerate(agent_response["messages"]):
                        if i==0 or i==len(agent_response["messages"])-1:
                            continue

                        if isinstance(re, AIMessage):
                            if re.content:
                                st.info(f"Agent: {re.content}")
                            if re.tool_calls:
                                for tool_call in re.tool_calls:
                                    st.info(f"Agent: {tool_call['name']}, {tool_call['args']}")
                        # elif isinstance(re, ToolMessage):
                        #     st.info(f"Tool: {re.content}")

                result = agent_response["messages"][-1].content
                logger.info(f"result: {result}")

            # st.info(f"Agent: {result}")

            st.markdown(result)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

            return result

def run_agent(query, historyMode, st):
    result = asyncio.run(mcp_rag_agent_multiple(query, historyMode, st))
    #result = asyncio.run(mcp_rag_agent_single(query, historyMode, st))

    logger.info(f"result: {result}")
    return result, [], []
