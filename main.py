import os
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI initialization
app = FastAPI()

# Define a request model for the query
class QueryRequest(BaseModel):
    query: str

# Function to get default start and end dates (past one year from today)
def get_default_dates():
    today = datetime.today()
    last_year = today - timedelta(days=365)
    return last_year.isoformat()[:10], today.isoformat()[:10]

# Function to reformat the user's query for the model
def reformat_query_for_model(user_query):
    start_date, end_date = get_default_dates()

    # Reformat the user's query, asking the model to extract information and return it in JSON format
    reformatted_query = f"""
    User Query: "{user_query}"

    Please extract the company name (entity), performance metric (parameter), and time range. 
    If the start date or end date is missing, assume the date range is from {start_date} to {end_date}.
    
    Return the result in the following JSON format:
    {{
        "entity": "<company_name>",
        "parameter": "<metric_name>",
        "startDate": "<start_date_iso>",
        "endDate": "<end_date_iso>"
    }}
    """
    return reformatted_query

# Function to call OpenAI GPT to process the query
def query_openai_model(reformatted_query):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # Or use "gpt-4" if available
        prompt=reformatted_query,
        max_tokens=150
    )
    return response['choices'][0]['text']  # Extract the generated text

# Function to process the user's query and extract metrics
def process_query(query):
    # Reformat the query for the model
    reformatted_query = reformat_query_for_model(query)
    
    # Send the reformatted query to the model and get the response
    model_response = query_openai_model(reformatted_query)

    # The model's response should already be in JSON format or close to it, so we can return it
    return model_response

# FastAPI route to handle POST requests
@app.post("/extract-metrics")
async def extract_metrics(request: QueryRequest):
    query = request.query
    result = process_query(query)
    return result
