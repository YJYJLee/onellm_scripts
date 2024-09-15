import json
import pandas as pd    

# Opening JSON file
jsonObj = pd.read_json(path_or_buf="/fsx-onellm/mingdachen/onellm-eval-data-tok/vizwiz/test.jsonl", lines=True)

# # returns JSON object as 
# # a dictionary
# data = json.load(jsonObj)

# Iterating through the json
# list
for i in jsonObj['question']:
    print(i)
