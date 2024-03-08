import requests

test_image = '/datalakes/0000/rapid_kit_custom/test/images/20220426092235763_jpeg_jpg.rf.f95f0eee0ea9e064a2ec9fbeff282ced.jpg'

url = 'http://0.0.0.0:8000/upload/'  # FastAPI 애플리케이션의 엔드포인트 URL
files = {'file': open(test_image, 'rb')}  # 업로드할 이미지 파일
response = requests.post(url, files=files)
print(response.json())

url = 'http://0.0.0.0:8000/generate/'  # FastAPI 애플리케이션의 엔드포인트 URL
response = requests.get(url)
print(response.json())
