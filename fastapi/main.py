from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse,JSONResponse
import base64
import shutil
import os
from func import *

# 이미지를 저장할 디렉터리 지정
STRIP_READER_DIR = "strip_reader/"
os.makedirs(STRIP_READER_DIR, exist_ok=True)
 
app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Backup existing files in the 'uploads/' directory
    backup_dir = '/datalakes/0001/rapid_kit/uploads/'
    os.makedirs(backup_dir, exist_ok=True)

    # Save the new file in the 'uploads/' directory
    with open('uploads/' + file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save the new file in the backup directory
    with open(backup_dir + file.filename, "wb") as buffer:
        file.file.seek(0)
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}

@app.get("/clear/")
async def clear_upload_dir():
    # Backup existing files in the 'uploads/' directory
    existing_files = os.listdir('uploads/')
    for existing_file in existing_files:
        os.remove('uploads/' + existing_file)
    return {"message": "upload directory is now empty"}

@app.post("/count/")
async def kit_count(request: Request):
    data = await request.body()
    target_dir = data.decode('utf-8')

    backup_dir = '/datalakes/0001/rapid_kit/count_outputs/'
    os.makedirs(backup_dir, exist_ok=True)
    existing_files = os.listdir('count_outputs')
    for existing_file in existing_files:
        if existing_file[0] != '.':
            shutil.rmtree('count_outputs/')

    print(f'target_dir : {target_dir}')
    kit_num_list = count_kit(base_path=target_dir)
    print(kit_num_list)
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, "count_outputs")
    
    img_files = []
    # 디렉토리 순회
    for root, dirs, files in os.walk(output_path):
        for file in files:
            file_path = os.path.join(root, file)
            img_files.append(file_path)
    img_files.sort()
    print(img_files)
    responses = []
    responses.append(kit_num_list)
    for image_file in img_files:
        with open(image_file, 'rb') as f:
            base64image = base64.b64encode(f.read())
            responses.append(base64image)
    if len(responses) != 0:
        return responses
    else:
        # 이미지가 없는 경우 예외처리 등을 수행할 수 있습니다.
        return {"message": "Image not found"}

@app.post("/generate/")
async def generate_result(request: Request):
    data = await request.body()
    target_dir = data.decode('utf-8')

    backup_dir = '/datalakes/0001/rapid_kit/outputs/'
    os.makedirs(backup_dir, exist_ok=True)
    existing_files = os.listdir('outputs/')
    for existing_file in existing_files:
        os.remove('outputs/' + existing_file)

    print(f'target_dir : {target_dir}')
    gen_result(base_path=target_dir)
    
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, "outputs")
    
    img_files = []
    # 디렉토리 순회
    for root, dirs, files in os.walk(output_path):
        for file in files:
            file_path = os.path.join(root, file)
            img_files.append(file_path)
    img_files.sort()
    print(img_files)
    responses = []
    for image_file in img_files:
        with open(image_file, 'rb') as f:
            base64image = base64.b64encode(f.read())
            responses.append(base64image)
    if len(responses) != 0:
        return responses
    else:
        # 이미지가 없는 경우 예외처리 등을 수행할 수 있습니다.
        return {"message": "Image not found"}
   
   
@app.post("/image_upload")
async def upload_image(request: Request):
    try:
        # 요청으로부터 바디를 읽음
        image_data = await request.body()
        file_location = f"{STRIP_READER_DIR}/image.jpg" # 파일명을 고정하거나 동적으로 생성할 수 있음
 
        # 이미지 데이터를 파일로 저장
        with open(file_location, "wb") as file_object:
            file_object.write(image_data)  # 파일 쓰기
 
        return JSONResponse(content={"message": f"File saved at '{file_location}'"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"An error occurred: {str(e)}"}, status_code=500)
     
@app.post("/image_upload2")
async def upload_image(file: UploadFile):
    try:
        content = await file.read()
        filename = "image.jpg"
        with open(os.path.join(STRIP_READER_DIR, filename), "wb") as fp:         
            fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

        return JSONResponse(content={"message": "File saved at '/image.jpg'"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"An error occurred: {str(e)}"}, status_code=500)