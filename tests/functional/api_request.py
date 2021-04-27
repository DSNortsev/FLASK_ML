import os
import requests
from utils import read_json

# Declare file path
json_data = '../data/data.json'
data = read_json(json_data)

# url = 'http://localhost:5000/predict'
url = 'http://0.0.0.0:1313/predict'
r = requests.post(url, headers={"content-type": "application/json"}, json=data)
print(r.json())
print(r.status_code)

