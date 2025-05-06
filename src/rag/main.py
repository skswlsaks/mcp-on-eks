import json
import traceback
import boto3
import os

from botocore.config import Config

from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from urllib import parse
from pydantic import BaseModel, Field

from fastapi import FastAPI

app = FastAPI()

bedrock_region = os.environ.get('bedrock_region') if os.environ.get('bedrock_region') else "ap-northeast-2"
modelId = os.environ.get('modelId') if os.environ.get('modelId') else "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
model_type = os.environ.get('model_type') if os.environ.get('model_type') else "claude"
path = "sharing"

knowledge_base_id = ""
numberOfDocs = 3
s3_prefix = 'docs'
doc_prefix = s3_prefix+'/'


def get_chat():
    maxOutputTokens = 4096 # 4k
    print(f'LLM: bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}')

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
        "max_tokens": maxOutputTokens,
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": ["\n\nHuman:"]
    }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock,
        model_kwargs=parameters,
        region_name=bedrock_region
    )

    return chat


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


def get_retrieval_grader(chat):
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    structured_llm_grader = chat.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def grade_documents(question, documents):
    print(f"###### grade_documents ######")
    print(f"start grading...")

    filtered_docs = []

    # Score each doc
    llm = get_chat()
    retrieval_grader = get_retrieval_grader(llm)
    for i, doc in enumerate(documents):
        # print('doc: ', doc)

        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        # print("score: ", score)

        grade = score.binary_score
        # print("grade: ", grade)

        if grade.lower() == "yes": # Document relevant
            print(f"---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)

        else: # Document not relevant
            print(f"---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return filtered_docs

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)

    updated_docs = []
    print('length of relevant_docs:', len(docs))
    for doc in docs:
        if doc.page_content in contentList:
            print('duplicated!')
            continue
        contentList.append(doc.page_content)
        updated_docs.append(doc)
    length_updated_docs = len(updated_docs)

    if length_original == length_updated_docs:
        print('no duplication')
    else:
        print('length of updated relevant_docs: ', length_updated_docs)

    return updated_docs


def search_by_knowledge_base(keyword: str, top_k: int) -> str:
    print("###### search_by_knowledge_base ######")

    global contentList, knowledge_base_id
    contentList = []

    print('keyword: ', keyword)
    keyword = keyword.replace('\'','')
    keyword = keyword.replace('|','')
    keyword = keyword.replace('\n','')
    print('modified keyword: ', keyword)

    relevant_docs = []
    print("LLM knowledge base id ", knowledge_base_id)

    if knowledge_base_id:
        try:
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id=knowledge_base_id,
                retrieval_config={"vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "HYBRID"   # SEMANTIC
                }},
            )

            docs = retriever.invoke(keyword)
            print('length of docs: ', len(docs))
            # print('docs: ', docs)

            print('--> docs from knowledge base')
            for i, doc in enumerate(docs):

                content = f"{keyword}에 대해 조사한 결과는 아래와 같습니다.\n\n"
                if doc.page_content:
                    content = doc.page_content

                score = doc.metadata["score"]

                link = ""
                if "s3Location" in doc.metadata["location"]:
                    link = doc.metadata["location"]["s3Location"]["uri"] if doc.metadata["location"]["s3Location"]["uri"] is not None else ""

                    # print('link:', link)
                    pos = link.find(f"/{doc_prefix}")
                    name = link[pos+len(doc_prefix)+1:]
                    encoded_name = parse.quote(name)
                    # print('name:', name)
                    link = f"{path}/{doc_prefix}{encoded_name}"

                elif "webLocation" in doc.metadata["location"]:
                    link = doc.metadata["location"]["webLocation"]["url"] if doc.metadata["location"]["webLocation"]["url"] is not None else ""
                    name = "WEB"

                url = link
                # print('url:', url)

                relevant_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': name,
                            'score': score,
                            'url': url,
                            'from': 'RAG'
                        },
                    )
                )

        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)

    return relevant_docs


from pydantic import BaseModel, Field

class Event(BaseModel):
    knowledge_base_name: str = Field(description="Name of the knowledge base")
    keyword: str = Field(description="Search keyword")
    top_k: int = Field(description="Number of top results to return")


@app.post("/")
async def retrieve_knowledge_base(event: Event):

    knowledge_base_name = event.knowledge_base_name
    keyword = event.keyword
    top_k = event.top_k

    global knowledge_base_id
    # retrieve knowledge_base_id
    if not knowledge_base_id:
        try:
            client = boto3.client(
                service_name='bedrock-agent',
                region_name=bedrock_region
            )
            response = client.list_knowledge_bases(
                maxResults=5
            )
            print('(list_knowledge_bases) response: ', response)

            if "knowledgeBaseSummaries" in response:
                summaries = response["knowledgeBaseSummaries"]
                for summary in summaries:
                    if summary["name"] == knowledge_base_name:
                        knowledge_base_id = summary["knowledgeBaseId"]
                        print('knowledge_base_id: ', knowledge_base_id)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)

    docs = []

    relevant_docs = search_by_knowledge_base(keyword, top_k)  # retrieve
    filtered_docs = grade_documents(keyword, relevant_docs)  # grade documents
    filtered_docs = check_duplication(filtered_docs)  # check duplication
    docs = filtered_docs


    json_docs = []
    for doc in docs:
        print('doc: ', doc)

        json_docs.append({
            "contents": doc.page_content,
            "reference": {
                "url": doc.metadata["url"],
                "title": doc.metadata["name"],
                "from": doc.metadata["from"]
            }
        })

    print('json_docs: ', json_docs)

    return {
        'response': json.dumps(json_docs, ensure_ascii=False)
    }

@app.get("/healthcheck")
def healthcheck():
    return JSONResponse({"status": "ok"})