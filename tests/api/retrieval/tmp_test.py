import requests

url = "http://localhost:8000/retrieval/"
data = {"query": "example search query"}

response = requests.post(url, json=data)  # 데이터를 JSON 형식으로 본문에 담아 전송

print(response.json())



# response = requests.get(url, params=data)

# print(response.json())