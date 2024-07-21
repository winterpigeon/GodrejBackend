from pyngrok import ngrok
from utils.rag import answerQuery
from utils.utils import images_and_summarize
from utils.finalSummary import create_rolling_summary
import chromadb
from typing import Dict
import uuid
import tempfile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

router = APIRouter()

client = chromadb.Client()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust this for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc)
    allow_headers=["*"],  # Allow all headers
)


class QueryRequest(BaseModel):
    query: str

# get product from query


@router.post("/uploadFile")
async def create_upload_file(file: UploadFile):

    print("recieved request")

    file_extension = file.filename.split(
        '.')[-1] if '.' in file.filename else 'tmp'

    with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_filename = tmp_file.name

    print(tmp_filename)

    print("getting page_summaries...")
    page_summaries = images_and_summarize(tmp_filename, client)
    print("got page_summaries!")
    print("getting final summary....")
    final_summary = create_rolling_summary(page_summaries)
    print("got final summary!")

    os.remove(tmp_filename)

    # For demonstration, let's just return the filename and some string
    return JSONResponse(content={"summary": final_summary})

# get product from query


@router.post("/answerQuery")
async def answer_query(request: QueryRequest):

    ans = answerQuery(request.query, client)
    return JSONResponse(content={"answer": ans})

app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
