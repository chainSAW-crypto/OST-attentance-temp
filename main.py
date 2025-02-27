import sys
print(sys.executable)

from pydantic import BaseModel
from typing import Optional, List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from Dependancies import ContextAwareChain, ReportSummarySystem
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
import os
from pathlib import Path
import re
from pathlib import Path
from langchain_core.documents import Document as LangChainDocument
from langchain.chains import load_summarize_chain
from langchain.schema import Document
from typing import List, Dict
import copy
from datetime import datetime
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException

app = FastAPI()
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Loading splitted documents and creating the retriever
with open('document_splits.pkl', 'rb') as f:
    loaded_splits = pickle.load(f)

vectorstore = Chroma.from_documents(documents=loaded_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class RequestData(BaseModel):
    company_info: str
    model_choice: Optional[str] = "llama-3.1-8b-instant"


# Response model
class ProcessingResponse(BaseModel):
    initial_summary: str
    model_used: str


@app.post("/api/process", response_model=ProcessingResponse)
async def process_company_data(data: RequestData):
    try:
        # Initialize API key and models
        api_key = os.getenv("GROQ_API_KEY")  # Replace with your actual API key handling

        model1 = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
        model2 = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

        # Initial summary generation
        summary_messages = [
            SystemMessage(
                content= "You are a smart summarizer, given the answers from questionnaire about any company you can efficiently summarize the content while retaining all important and significant information about the company"),
            HumanMessage(content=f"Hey, this is the broad information about the company= {data.company_info}")
        ]

        summary_content = model1(summary_messages).content

        # Create context-aware chain
        context_chain = ContextAwareChain(model1, retriever, summary_content)

        # Get specialized chains
        strategy_chain = context_chain.get_strategy_chain()
        finance_chain = context_chain.get_finance_chain()
        marketing_chain = context_chain.get_marketing_chain()

        # Process through different chains
        strategy_response = strategy_chain({
            "input": "Please develop a comprehensive corporate strategy."
        })

        finance_response = finance_chain({
            "input": "Please provide a financial strategy for our company."
        })

        marketing_response = marketing_chain({
            "input": "Please create a marketing strategy for our company."
        })

        # Compile full report
        full_report = [
            marketing_response['response'],
            finance_response['response'],
            strategy_response['response']
        ]

        # Create summary system and generate initial summary
        summary_system = ReportSummarySystem(model2)
        initial_summary = summary_system.create_initial_summary(full_report)

        # Return response
        return ProcessingResponse(
            initial_summary=initial_summary,
            model_used=data.model_choice
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during processing: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}