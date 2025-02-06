# Remove medical imports and add financial ones
from fastapi import FastAPI, Request, HTTPException, Response, UploadFile, File  # Added for voice command support
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import os
import json
import sys
from typing import Optional, Dict, Any
import tempfile
import speech_recognition as sr  # Import library for voice input

# Remove: sys.path.insert(0, r"F:\Wearables\Medical-RAG-LLM\Data")

# Remove: from insurance_data import insurance_data
from pydantic import BaseModel

# Extend the query model to include conversation history and language
class QueryRequest(BaseModel):
    query: str
    conversation_context: Optional[str] = None  # For conversational context
    language: Optional[str] = "English"  # Specify language of the query

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize LLM config
config = {
    'max_new_tokens': 256,  # Reduced for faster responses
    'context_length': 512,  # Reduced context window
    'temperature': 0.3,  # More focused responses
    'top_p': 0.95,
    'stream': False,
    'threads': min(4, int(os.cpu_count() / 2)),
}

# Initialize the financial query prompt template
FINANCIAL_QUERY_PROMPT = """
Answer the query using only the provided context information. Be direct and concise.

Context: {context}
Query: {query}

Response:"""

# Introduce a new prompt specifically handling MaxLife vs. LIC comparisons
COMPARISON_PROMPT = """
Compare the products/policies based only on the information in the context. List key differences.

Context: {context}
Query: {query}

Key differences:"""

# Update model path
MODEL_PATH = "F:/Wearables/Medical-RAG-LLM/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Initialize components
try:
    # Use local model directly
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config=config
    )
    print("Successfully loaded local model from:", MODEL_PATH)
    
    # Initialize embeddings with specific kwargs
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    from qdrant_client.http.models import VectorParams  # add this import

    # Initialize Qdrant client
    client = QdrantClient("http://localhost:6333")

    # Simply connect to the collection, don't try to create it
    try:
        # Create vector store for financial documents
        db = Qdrant(
            client=client, 
            embeddings=embeddings,
            collection_name="financial_docs"
        )
        print("Connected to 'financial_docs' collection")
        
        retriever = db.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"Error connecting to collection: {e}")
        raise
    
except Exception as e:
    print(f"Initialization error: {e}")
    print(f"Make sure the model exists at: {MODEL_PATH}")
    raise

# Add a simple intent detection utility
def detect_intent(query: str) -> str:
    """Perform rule-based intent identification with minimal latency."""
    lowered = query.lower()
    if "compare" in lowered or ("maxlife" in lowered and "lic" in lowered):
        return "comparison"
    # Could add more rules for follow-up or clarifications if needed
    return "standard"

# New main endpoint to handle financial queries (used by the frontend)
@app.post("/query_new")
async def process_query_new(request: QueryRequest):
    """Handle financial queries with intent, conversational context, and language matching"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # Get most relevant chunks with higher similarity threshold
        docs = retriever.get_relevant_documents(
            query,
            search_type="mmr",  # Use MMR for better diversity in results
            search_kwargs={"k": 3, "fetch_k": 5}  # Fetch more, return best
        )
        
        if not docs:
            return JSONResponse(content={
                "query": query,
                "response": "I don't have enough information to answer that question accurately."
            })

        # Consolidate context more effectively
        context_parts = []
        for doc in docs:
            content = doc.page_content.strip()
            if content:
                context_parts.append(content)
        
        context = " ".join(context_parts)
        
        # Use shorter context for faster processing
        context = context[:500]
        
        # Select appropriate prompt
        intent = detect_intent(query)
        prompt = COMPARISON_PROMPT if intent == "comparison" else FINANCIAL_QUERY_PROMPT
        
        # Add language instruction if specified
        if request.language and request.language.lower() != "english":
            prompt += f"\nRespond in {request.language}."
            
        # Generate response with timeout
        response = llm(
            prompt.format(context=context, query=query),
            max_tokens=256,
            temperature=0.3
        )
        
        if not response or response.isspace():
            return JSONResponse(content={
                "query": query,
                "response": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            })
            
        return JSONResponse(content={
            "query": query,
            "response": response.strip()
        })
        
    except Exception as e:
        print(f"Error in query processing: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Please try again."
        )

# New alias endpoint to support legacy POST requests to "/query"
@app.post("/query")
async def query_alias(request: QueryRequest):
    return await process_query_new(request)

# Add a new endpoint for voice commands
@app.post("/query_voice")
async def query_voice(file: UploadFile = File(...), conversation_context: Optional[str] = None, language: Optional[str] = "English"):
    """Convert voice to text then process query"""
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_name = tmp.name
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_name) as source:
            audio_data = recognizer.record(source)
            language_code = "en-US" if language.lower() == "english" else None
            text = recognizer.recognize_google(audio_data, language=language_code) if language_code else recognizer.recognize_google(audio_data)
        query_request = QueryRequest(query=text, conversation_context=conversation_context, language=language)
        return await process_query_new(query_request)
    except Exception as e:
        print(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")

# Add health-check endpoint
@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Add a route to handle favicon.ico requests
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Add helper function to search financial info
def search_financial_info(query: str) -> dict:
    """Search through financial documents"""
    results = []
    query = query.lower()
    
    # Removed insurance_data search block since the file does not exist.
    
    # Add results from vector store
    docs = retriever.invoke(query)
    for doc in docs:
        results.append({
            "type": "document",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown")
        })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)