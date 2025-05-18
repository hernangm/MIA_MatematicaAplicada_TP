import json

def load(filePath): 
    with open(filePath) as file:
        return json.load(file)