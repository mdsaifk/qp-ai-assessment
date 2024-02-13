from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA ,ConversationalRetrievalChain ,LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import sys
import os
import textwrap
from typing import Union
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

persist_directory = 'chroma_db_vectorstore'

def load_embeddings(persist_directory):
    embedding = OpenAIEmbeddings()
    doc_with_embedding = Chroma(embedding_function = embedding,persist_directory= persist_directory)
    return doc_with_embedding


class Item(BaseModel):
    user_question: str

# def generate_response(chat_history, user_question):
#     genai.configure(api_key="gemini api key")  # Replace 'your-api-key' with your actual API key

#     generation_config = {
#         "temperature": 0,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 256,
#     }

#     safety_settings = [
#         {
#             "category": "HARM_CATEGORY_HARASSMENT",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_HATE_SPEECH",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         },
#         {
#             "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#             "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#         }
#     ]

#     model = genai.GenerativeModel(model_name="gemini-pro",
#                                   generation_config=generation_config,safety_settings=safety_settings)

    # prompt1_1 = f"Given the following last conversation log, if user question is related to previous user question then formulate a question that would be the most relevant to provide the user with an answer otherwise don't formulate question leave it as it is.\n\nCONVERSATION LOG: \n{chat_history}\n\nuser question: {user_question}\n\nRefined user question :"
    
    # prompt_parts = [prompt1_1,]
    # response = model.generate_content(prompt_parts)
    # return response.text

chat_history = ["","","",""]
@app.post("/ask", response_model=Union[dict, None])
async def ask(request: Request, item: Item):
 
    user_question = item.user_question
    doc_with_embedding = load_embeddings(persist_directory)

    # new_query = user_question
    retriever = doc_with_embedding.as_retriever(search_type="similarity", search_kwargs={"k":3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key = api_key,temperature=0)  
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    

    qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever,memory = memory )      #return_source_documents=True

    with get_openai_callback() as cb:
        prompt2 = f"""You're a helpful, respectful, and honest assistant. 
            Answer the user question truthfully and shortly using the provided context and rephrase it better. If the answer is not in the context,just say I don't know the answer
            and avoid creating the answer by yourself.\n\n
            Question: {user_question}\n\n"""
        
        
        result = qa({"query": prompt2, "chat_history": chat_history})
        response_text = result["result"]
        chat_history.extend(('user_question:' + user_question, 'Assistant :' + response_text))
        # print(cb)
    return {"response": response_text}

if __name__ == '__main__':
    uvicorn.run(app)



#check the response from chatbot at this url http://localhost:8000/docs#/default/ask_ask_post
    

#uvicorn Retrievalqa:app --reload --host 0.0.0.0 --port 1000