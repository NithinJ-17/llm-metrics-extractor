import os
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

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
    """
    Returns the default start and end date as ISO 8601 format.
    Start date: One year from today
    End date: Today's date
    """
    today = datetime.today()
    last_year = today - timedelta(days=365)
    return last_year.isoformat()[:10], today.isoformat()[:10]

# Function to reformat the user's query for the model
def reformat_query_for_model(user_query):
    """
    Reformat the user's query to instruct the language model (LLM) to return the response
    in JSON format, extracting company names (entities), performance metrics, and dates.
    If no dates are provided, it uses the default date range.
    
    Args:
    user_query (str): The natural language query from the user.
    
    Returns:
    str: Reformatted query to be sent to the LLM.
    """
    start_date, end_date = get_default_dates()

    # Reformat the user's query for multiple entities and to extract relevant data
    reformatted_query = f"""
    User Query: "{user_query}"

    Please extract all company names (entities), performance metrics (parameters), and time ranges.
    If the start date or end date is missing, assume the date range is from {start_date} to {end_date}.

    Return the result in the following JSON format, with multiple companies if applicable:
    [
        {{
            "entity": "<company_name>",
            "parameter": "<metric_name>",
            "startDate": "<start_date_iso>",
            "endDate": "<end_date_iso>"
        }},
        ...
    ]
    """
    return reformatted_query

# Function to call OpenAI GPT to process the query
def query_openai_model(reformatted_query):
    """
    Sends the reformatted query to OpenAI GPT model and returns the generated response.
    
    Args:
    reformatted_query (str): The reformatted query to be processed by GPT.

    Returns:
    str: Generated text from the model.
    """
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Or use "gpt-4" if available
            prompt=reformatted_query,
            max_tokens=150
        )
        return response['choices'][0]['text']  # Extract the generated text
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# Function to validate the model's response to ensure it is in valid JSON format
def validate_response(model_response):
    """
    Validates the model's response to ensure it is a valid JSON object or list of objects.
    
    Args:
    model_response (str): The response from the model.

    Returns:
    dict or list: Parsed JSON data.
    
    Raises:
    ValueError: If the model response is not valid JSON or is missing required fields.
    """
    try:
        # Try parsing the model's response as JSON
        parsed_response = json.loads(model_response)
        
        # Check if it is a list and has required fields for each entry
        if isinstance(parsed_response, list):
            for item in parsed_response:
                if not all(key in item for key in ["entity", "parameter", "startDate", "endDate"]):
                    raise ValueError("Response is missing required fields")
            return parsed_response
        else:
            raise ValueError("Expected a list of JSON objects")
    
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON")

# Function to process the user's query and extract metrics
def process_query(query):
    """
    Processes the user's query, sends it to the LLM, and validates the response.
    
    Args:
    query (str): The user's query in natural language.
    
    Returns:
    dict or list: Structured JSON response from the model.
    
    Raises:
    HTTPException: For validation errors or API errors.
    """
    try:
        # Reformat the query for the model
        reformatted_query = reformat_query_for_model(query)
        
        # Send the reformatted query to the model and get the response
        model_response = query_openai_model(reformatted_query)

        # Validate and parse the response
        validated_response = validate_response(model_response)
        return validated_response
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

# FastAPI route to handle POST requests
@app.post("/extract-metrics")
async def extract_metrics(request: QueryRequest):
    """
    API endpoint to extract metrics from a user query.
    
    Args:
    request (QueryRequest): The user query sent via POST request.
    
    Returns:
    JSON: Extracted company names, performance metrics, and time range in JSON format.
    """
    query = request.query
    result = process_query(query)
    return result
