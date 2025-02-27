from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class UpdateRequest(BaseModel):
    existing_report: str
    user_feedback: str


class UpdateResponse(BaseModel):
    updated_report: str


# Initialize the Groq model and prompt template
def initialize_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    update_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant tasked with updating an existing report based on user feedback. 
        The report should retain its original structure and context while incorporating the feedback provided 
        by the user. Your task is to update the report text by integrating the feedback in a way that makes 
        the report more accurate, relevant, and useful. Do not include your thinking in the response; only 
        return the updated report. Keep the token size within 4000 tokens."""),
        ("human", "Current Report: {report}\nUser Feedback: {feedback}"),
    ])

    model = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    return update_prompt | model


# Initialize the chain
update_chain = initialize_chain()


@app.post("/update_report", response_model=UpdateResponse)
async def update_report(request_data: UpdateRequest):
    # Validate input
    if not request_data.existing_report or not request_data.user_feedback:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields. Please provide 'existing_report' and 'user_feedback'"
        )

    try:
        # Process the update
        response = update_chain.invoke({
            "report": request_data.existing_report,
            "feedback": request_data.user_feedback
        })

        return UpdateResponse(updated_report=response.content)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during processing: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    # Check for API key on startup
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable must be set")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)