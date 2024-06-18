
import requests
API_TOKEN="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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/statistics?dataset=google/fleurs&config=es_419&split=validation"
API_URL = "https://datasets-server.huggingface.co/statistics?dataset=openai/openai_humaneval&config=openai_humaneval&split=test"
API_UTL = "https://datasets-server.huggingface.co/statistics?dataset=google-research-datasets/mbpp&config=full&split=test"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()

print(data)