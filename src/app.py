import requests

url = 'http://raspberrypi.local:5000/classify'

data = {}

response = requests.post(url, json=data)
print(response.text)