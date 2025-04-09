import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
import json
import requests
import shutil
import logging
import cv2
import numpy as np
import pymupdf
from qreader import QReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.ocr.extractor import extract_tables_from_file, allowed_file
from src.llm.processor import process_data_with_llm
from config.settings import SYSTEM_TEMPLATE

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Combined QR and OCR Processing API")

# Пути для временных файлов
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Инициализация QReader
qreader = QReader()

# Допустимые расширения файлов
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}

# Функции для QR-кодов (из второго сервиса)
def extract_images_from_pdf(pdf_path, output_folder="temp_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_paths = []
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        doc = pymupdf.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images()
            for image_index, img in enumerate(image_list, start=1):
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                output_filename = f"{pdf_filename}_page_{page_index+1}_image_{image_index}.png"
                output_path = os.path.join(output_folder, output_filename)
                pix.save(output_path)
                image_paths.append(output_path)
                pix = None
        doc.close()
    except Exception as e:
        logger.error(f"Ошибка извлечения изображений из PDF: {str(e)}")
        raise
    return image_paths

def decode_qr_codes_from_images(image_folder="temp_images"):
    qr_results = []
    for image_file in os.listdir(image_folder):
        if not image_file.lower().endswith('.png'):
            continue
        image_path = os.path.join(image_folder, image_file)
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            decoded_text = qreader.detect_and_decode(image=image)
            if decoded_text and decoded_text[0]:
                qr_results.append(decoded_text[0])
        except Exception as e:
            logger.warning(f"Ошибка декодирования QR-кода из {image_file}: {str(e)}")
    return qr_results

def decode_qr_code_from_single_image(image_path):
    qr_results = []
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Не удалось загрузить изображение из {image_path}")
            return qr_results
        decoded_text = qreader.detect_and_decode(image=image)
        if decoded_text and decoded_text[0]:
            qr_results.append(decoded_text[0])
    except Exception as e:
        logger.warning(f"Ошибка декодирования QR-кода из {image_path}: {str(e)}")
    return qr_results

def send_post_request(qr_data):
    url = "https://proverkacheka.com/api/v1/check/get"
    payload = {
        "token": os.getenv("API_TOKEN"),  
        "qrraw": qr_data
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        logger.error(f"Ошибка отправки POST-запроса для QR-данных {qr_data}: {str(e)}")
        return None

# Основной эндпоинт
@app.post("/process-file/", response_model=dict)
async def process_file(file: UploadFile = File(...)):
    """
    Принимает PDF или изображение, сначала проверяет QR-код,
    затем извлекает таблицы и текст через OCR + LLM, если QR-кода нет.
    """
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла. Разрешены: .pdf, .png, .jpg, .jpeg")

    input_file_path = INPUT_DIR / file.filename
    with open(input_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    temp_output_path = OUTPUT_DIR / f"temp_{file.filename}.json"
    file_extension = os.path.splitext(file.filename)[1].lower()

    try:
        # Шаг 1: Проверка QR-кода
        qr_data_list = []
        if file_extension == '.pdf':
            image_paths = extract_images_from_pdf(str(input_file_path))
            qr_data_list = decode_qr_codes_from_images()
        else:
            qr_data_list = decode_qr_code_from_single_image(str(input_file_path))

        # Если QR-код найден, отправляем запрос и возвращаем результат
        if qr_data_list:
            responses = []
            for qr_data in qr_data_list:
                response = send_post_request(qr_data)
                if response:
                    responses.append(response)
            if responses:
                return responses[0]  # Возвращаем первый успешный результат (можно настроить)

        # Шаг 2: Если QR-кода нет, используем OCR + LLM
        extracted_json_path = extract_tables_from_file(str(input_file_path), str(temp_output_path))
        structured_json = process_data_with_llm(extracted_json_path, SYSTEM_TEMPLATE)
        return json.loads(structured_json)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        # Очистка временных файлов
        if file_extension == '.pdf' and os.path.exists("temp_images"):
            shutil.rmtree("temp_images")
        if input_file_path.exists():
            os.remove(input_file_path)
        if temp_output_path.exists():
            os.remove(temp_output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# check