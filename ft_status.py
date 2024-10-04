from openai import OpenAI 
from dotenv import load_dotenv
import os

# Ensure your API key is set
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)



# List recent fine-tuning jobs (e.g., latest 5 jobs)
response = client.fine_tuning.jobs.list(limit=5)
print(response)
# for job in response['data']:
#     print(f"Job ID: {job['id']}, Status: {job['status']}")