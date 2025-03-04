from fastapi import FastAPI
from pydantic import BaseModel
import openai
from fastapi.middleware.cors import CORSMiddleware

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

# Create FastAPI app
app = FastAPI()

# CORS configuration: Define allowed origins, methods, headers
origins = [
    "http://localhost",  # If you're hosting the frontend locally
    "http://localhost:8000",  # FastAPI server's port
    "http://127.0.0.1:8000",  # Common address for local development
]

# Add CORSMiddleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allowed origins (front-end domains)
    allow_credentials=True,  # Allow credentials (cookies, etc.)
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers (you can specify certain ones if needed)
)

# Define the message schema for chat
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message
    
    # Call OpenAI GPT-3 or your chatbot model here
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use your model of choice
        prompt=user_message,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    chatbot_reply = response.choices[0].text.strip()
    return {"response": chatbot_reply}
