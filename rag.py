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
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import speech_recognition as sr  # Import library for voice input
from enum import Enum
import re
from config.models import ModelProvider, MODEL_CONFIGS, REPORT_TEMPLATE
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import anthropic
import groq
import openai
import together

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

# Update config with more stable parameters
config = {
    'max_new_tokens': 512,  # Increased for better completion
    'context_length': 2048,  # Increased context window
    'temperature': 0.7,  # More creative responses
    'top_p': 0.95,
    'top_k': 50,  # Added for better token selection
    'stream': False,
    'threads': min(4, int(os.cpu_count() / 2)),
}

# Simplified prompt templates
FINANCIAL_QUERY_PROMPT = """Use the following context to answer the question:
Context: {context}
Question: {query}
Answer: Let me help you with that."""

COMPARISON_PROMPT = """Compare the following based on the context provided:
Context: {context}
Question: {query}
Comparison: Let me compare these for you."""

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

# Add new intent classification system
class IntentType(Enum):
    COMPARISON = "comparison"
    PRODUCT_INFO = "product_info"
    COST_ANALYSIS = "cost_analysis"
    COVERAGE_DETAILS = "coverage_details"
    ELIGIBILITY = "eligibility"
    CLAIM_PROCESS = "claim_process"
    STANDARD = "standard"

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            IntentType.COMPARISON: [
                r"compare|versus|vs|difference|better|which is|compare between",
                r"(maxlife|lic).*(maxlife|lic)",
                r"which (policy|plan|insurance) (is|would be) better"
            ],
            IntentType.PRODUCT_INFO: [
                r"what (is|are) .*(policy|plan|insurance|coverage)",
                r"tell me about|explain|describe",
                r"features|benefits|details"
            ],
            IntentType.COST_ANALYSIS: [
                r"cost|price|premium|fee|charge|expensive|cheaper",
                r"how much|payment|monthly|annually",
                r"budget|affordable"
            ],
            IntentType.COVERAGE_DETAILS: [
                r"cover|coverage|protect|benefit|claim",
                r"what (does|do|will) .* cover",
                r"maximum|minimum|limit"
            ],
            IntentType.ELIGIBILITY: [
                r"eligible|qualify|who can|requirement",
                r"criteria|condition|age limit",
                r"can I|should I"
            ],
            IntentType.CLAIM_PROCESS: [
                r"claim|process|procedure|file|submit",
                r"how (to|do|can) I claim",
                r"settlement|payout"
            ]
        }
        
    def _match_patterns(self, text: str, patterns: List[str]) -> float:
        text = text.lower()
        matches = sum(1 for pattern in patterns if re.search(pattern, text))
        return matches / len(patterns) if matches > 0 else 0

    def classify(self, query: str) -> Tuple[IntentType, float]:
        max_score = 0
        intent = IntentType.STANDARD
        
        for intent_type, patterns in self.intent_patterns.items():
            score = self._match_patterns(query, patterns)
            if score > max_score:
                max_score = score
                intent = intent_type
        
        return intent, max_score

# Initialize intent classifier
intent_classifier = IntentClassifier()

# Update the detect_intent function
def detect_intent(query: str) -> str:
    """Enhanced intent detection with confidence scoring"""
    intent, confidence = intent_classifier.classify(query)
    
    # Log intent detection for monitoring
    print(f"Intent: {intent.value}, Confidence: {confidence:.2f}, Query: {query}")
    
    # Use specific prompts based on intent
    if confidence >= 0.3:  # Confidence threshold
        return intent.value
    return "standard"

# Update prompt templates for each intent
INTENT_PROMPTS = {
    IntentType.COMPARISON.value: """Compare these financial products based on the context:
Context: {context}
Question: {query}
Focus on key differences in features, costs, and benefits.
Comparison:""",

    IntentType.PRODUCT_INFO.value: """Explain this financial product based on the context:
Context: {context}
Question: {query}
Focus on main features and benefits.
Response:""",

    IntentType.COST_ANALYSIS.value: """Analyze the costs based on the context:
Context: {context}
Question: {query}
Focus on pricing, premiums, and payment terms.
Analysis:""",
}

class ModelManager:
    def __init__(self):
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        for model_name, config in MODEL_CONFIGS.items():
            if config["provider"] == ModelProvider.LOCAL:
                self.models[model_name] = self._init_local_model(config)
            elif config["provider"] == ModelProvider.GROQ:
                self.models[model_name] = self._init_groq_model(config)
            # ... Initialize other providers similarly

    def _init_local_model(self, config):
        return CTransformers(
            model=config["model_path"],
            model_type=config["model_type"],
            config=config["config"]
        )

    def _init_groq_model(self, config):
        return groq.Groq(api_key=config["api_key"])

    async def generate_response(self, model_name: str, prompt: str, **kwargs):
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        config = MODEL_CONFIGS[model_name]
        if config["provider"] == ModelProvider.LOCAL:
            return model(prompt, **kwargs)
        elif config["provider"] == ModelProvider.GROQ:
            response = await model.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config["model_id"],
                **config["config"]
            )
            return response.choices[0].message.content
        # ... Handle other providers

# Initialize model manager
model_manager = ModelManager()

# Update the retriever configuration for better results
@app.post("/query_new")
async def process_query_new(
    request: QueryRequest,
    model_name: str = "local-mistral"
):
    try:
        query = request.query.strip()
        if not query:
            return JSONResponse(content={
                "query": "",
                "response": "Please enter a query"
            })

        try:
            # Get documents with simpler retrieval
            docs = retriever.get_relevant_documents(query)
            if not docs:
                return JSONResponse(content={
                    "query": query,
                    "response": "I don't have enough information to answer that question."
                })

            # Simplify context processing
            context = " ".join([
                doc.page_content.strip()
                for doc in docs[:2]  # Limit to top 2 docs
                if doc.page_content.strip()
            ])

            # Generate response with explicit error handling
            try:
                intent = detect_intent(query)
                prompt = INTENT_PROMPTS.get(intent, FINANCIAL_QUERY_PROMPT)
                full_prompt = prompt.format(context=context[:1024], query=query)
                
                response = await model_manager.generate_response(
                    model_name=model_name,
                    prompt=full_prompt,
                    **MODEL_CONFIGS[model_name]["config"]
                )
                
                # Store conversation history for report generation
                if not hasattr(request, "session"):
                    request.session = {}
                request.session.setdefault("conversation_history", []).append({
                    "query": query,
                    "response": response
                })
                
                if not response or not response.strip():
                    return JSONResponse(content={
                        "query": query,
                        "response": "I understand your question but couldn't generate a proper response. Please try rephrasing."
                    })
                    
                return JSONResponse(content={
                    "query": query,
                    "response": response.strip(),
                    "model": model_name
                })
                
            except Exception as e:
                print(f"LLM Generation Error: {str(e)}")
                return JSONResponse(content={
                    "query": query,
                    "response": "I encountered an issue while processing your query. Please try again."
                })
                
        except Exception as e:
            print(f"Document Retrieval Error: {str(e)}")
            return JSONResponse(content={
                "query": query,
                "response": "Error accessing the knowledge base. Please try again."
            })
            
    except Exception as e:
        print(f"General Error: {str(e)}")
        return JSONResponse(content={
            "query": query if 'query' in locals() else "",
            "response": "An unexpected error occurred. Please try again."
        })

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

# Add report generation endpoint
@app.post("/generate_report")
async def generate_report(
    request: Request,
    model_name: str = "local-mistral",
    format: str = "pdf"
):
    conversation_history = request.session.get("conversation_history", [])
    
    # Generate report content using the selected model
    report_prompt = REPORT_TEMPLATE.format(
        summary="Summarize our conversation",
        points="Extract key points",
        recommendations="Provide recommendations",
        next_steps="Suggest next steps",
        model_name=model_name,
        date=datetime.now().strftime("%Y-%m-%d")
    )
    
    report_content = await model_manager.generate_response(
        model_name=model_name,
        prompt=report_prompt
    )

    if format == "pdf":
        # Generate PDF using ReportLab
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Financial Consultation Report", title_style))
        story.append(Spacer(1, 12))

        # Add content
        for line in report_content.split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        
        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=financial_report.pdf"
            }
        )

    return JSONResponse(content={"report": report_content})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)