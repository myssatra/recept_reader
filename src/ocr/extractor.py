# src/ocr/extractor.py
from img2table.document import PDF, Image
from img2table.ocr import TesseractOCR
import os
from PIL import Image as PILImage
import numpy as np
import tempfile
import json
import pytesseract

ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}

def allowed_file(filename):
    """Проверяет, поддерживается ли расширение файла."""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def extract_text_outside_tables(image_path, tables):
    """Извлекает текст вне таблиц с помощью Tesseract."""
    try:
        img = PILImage.open(image_path)
        print(f"Открыто изображение: {image_path}, размер: {img.size}")
    except Exception as e:
        print(f"Ошибка открытия изображения {image_path}: {str(e)}")
        return []

    table_bboxes = [table.bbox for table in tables if table.bbox is not None]
    print(f"Найдено таблиц с bbox: {len(table_bboxes)}")

    # Используем psm=1 для извлечения текста вне таблиц
    full_text_data = pytesseract.image_to_data(img, lang="rus+eng", config='--psm 1', output_type=pytesseract.Output.DICT)
    outside_text = []
    for i in range(len(full_text_data['text'])):
        if full_text_data['text'][i].strip():
            x, y, w, h = (full_text_data['left'][i], full_text_data['top'][i],
                          full_text_data['width'][i], full_text_data['height'][i])
            text_bbox = (x, y, x + w, y + h)
            is_outside = True
            for table_bbox in table_bboxes:
                tx1, ty1, tx2, ty2 = table_bbox.x1, table_bbox.y1, table_bbox.x2, table_bbox.y2
                if not (text_bbox[2] < tx1 or text_bbox[0] > tx2 or text_bbox[3] < ty1 or text_bbox[1] > ty2):
                    is_outside = False
                    break
            if is_outside:
                outside_text.append({
                    "text": full_text_data['text'][i],
                    "bbox": [x, y, x + w, y + h],
                    "confidence": float(full_text_data['conf'][i])
                })
    print(f"Извлечено текста вне таблиц: {len(outside_text)} элементов")
    return outside_text

def extract_tables_from_file(file_path, output_path):
    """Извлекает таблицы и текст вне таблиц из файла (PDF или изображения) и сохраняет в JSON."""
    if not allowed_file(file_path):
        raise ValueError(f"Формат файла не поддерживается. Разрешённые форматы: {ALLOWED_EXTENSIONS}")

    if not os.path.exists(file_path):
        raise ValueError(f"Файл не найден: {file_path}")

    file_extension = os.path.splitext(file_path.lower())[1]

    all_data = {"tables": [], "text": []}

    # Единая настройка Tesseract с psm=1 для всех типов файлов
    tess_ocr = TesseractOCR(n_threads=1, lang="rus+eng", psm=1)
    print("Используется Tesseract с psm=1 для всех файлов")

    if file_extension == '.pdf':
        document = PDF(file_path, detect_rotation=True, pdf_text_extraction=False)
        num_pages = len(document.images)
        print(f"Количество страниц в PDF: {num_pages}")

        for page_idx, page_image in enumerate(document.images):
            print(f"Обработка страницы {page_idx + 1}/{num_pages}")
            temp_file_path = None
            try:
                pil_img = PILImage.fromarray(np.uint8(page_image))
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    pil_img.save(temp_file_path)

                page_doc = Image(src=temp_file_path)
                extracted_tables = page_doc.extract_tables(
                    ocr=tess_ocr,
                    implicit_rows=False,
                    implicit_columns=False,
                    borderless_tables=True,
                    min_confidence=50
                )
                print(f"Извлеченные таблицы со страницы {page_idx + 1}: {len(extracted_tables)}")

                for table_idx, table in enumerate(extracted_tables):
                    if hasattr(table, 'df') and table.df is not None:
                        table_data = {
                            "page": page_idx + 1,
                            "table_index": table_idx,
                            "data": table.df.to_dict(orient="records")
                        }
                        all_data["tables"].append(table_data)

                outside_text = extract_text_outside_tables(temp_file_path, extracted_tables)
                for text_item in outside_text:
                    text_item["page"] = page_idx + 1
                    all_data["text"].append(text_item)

            except Exception as e:
                print(f"Ошибка при обработке страницы {page_idx + 1}: {str(e)}")
                raise
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError as e:
                        print(f"Предупреждение: Не удалось удалить временный файл {temp_file_path}: {str(e)}")

    else:
        print(f"Обработка изображения: {file_path}")
        try:
            document = Image(file_path, detect_rotation=True)
            print(f"Изображение успешно загружено: {file_path}")
            extracted_tables = document.extract_tables(
                ocr=tess_ocr,
                implicit_rows=False,
                implicit_columns=False,
                borderless_tables=True,
                min_confidence=50
            )
            print(f"Извлеченные таблицы из изображения: {len(extracted_tables)}")

            for table_idx, table in enumerate(extracted_tables):
                if hasattr(table, 'df') and table.df is not None:
                    print(f"Таблица {table_idx} содержит данные: {table.df.shape}")
                    table_data = {
                        "table_index": table_idx,
                        "data": table.df.to_dict(orient="records")
                    }
                    all_data["tables"].append(table_data)
                else:
                    print(f"Таблица {table_idx} не содержит DataFrame")

            outside_text = extract_text_outside_tables(file_path, extracted_tables)
            all_data["text"].extend(outside_text)

        except Exception as e:
            print(f"Ошибка при обработке изображения {file_path}: {str(e)}")
            raise

    if not all_data["tables"] and not all_data["text"]:
        print("Не удалось извлечь ни таблицы, ни текст из файла")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Все таблицы и текст сохранены в {output_path}")

    return output_path