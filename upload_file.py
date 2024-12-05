import os
import yaml
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field
from PIL import Image
import pdfplumber
import docx
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from medical_record import process_text
from admission_medical_record import process_medical_text
from fastapi import APIRouter
from logger_config import logger
import numpy as np


# 读取配置文件
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


# 根据参数选择合适的文本处理函数
def get_processing_function(document_type: int):
    """
    根据传入的 document_type 返回相应的处理函数。
    """
    if document_type == 0:
        return process_text
    elif document_type == 1:
        return process_medical_text
    else:
        return process_text  # 未定义类型时的默认处理函数


# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', rec=True, det=False)

# 上传文件保存目录
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 用于处理不同格式的文件
def extract_text_from_pdf(filepath: str) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text.strip()


def extract_text_from_image(filepath: str) -> str:
    # 使用 PIL 打开图片
    image = Image.open(filepath)

    # 将 PIL 图像转换为 NumPy 数组

    image_np = np.array(image)

    # 使用 PaddleOCR 处理
    result = ocr.ocr(image_np, cls=True)

    extracted_text = ""
    for block in result:
        for line in block:
            text = line[1][0]
            extracted_text += text
    return extracted_text


def extract_text_from_word(filepath: str) -> str:
    doc = docx.Document(filepath)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text).strip()


def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().strip()


# 处理文件内容
def process_file(filepath: str) -> str:
    logger.info(f"正在处理文件: {filepath}")
    try:
        # 获取文件扩展名
        file_extension = os.path.splitext(filepath)[1].lower()

        # 根据文件类型选择提取方式
        if file_extension == '.pdf':
            logger.info("正在从PDF提取文本...")
            text = extract_text_from_pdf(filepath)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            logger.info("正在从图片提取文本...")
            text = extract_text_from_image(filepath)
        elif file_extension == '.docx':
            logger.info("正在从Word文档提取文本...")
            text = extract_text_from_word(filepath)
        elif file_extension == '.txt':
            logger.info("正在从文本文件提取文本...")
            text = extract_text_from_txt(filepath)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

        logger.info(f"文本提取成功，共提取了 {len(text)} 个字符。")
        return text
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}", exc_info=True)
        raise


router = APIRouter()


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), document_type: int = 0):
    """
    上传文件并根据 document_type 选择处理逻辑。
    :param file: 上传的文件
    :param document_type: 文档类型，用于选择处理逻辑
    :return: 结构化结果或错误信息
    """
    try:
        logger.info(f"收到文件: {file.filename}, 文档类型: {document_type}")

        # 检查文件是否为空
        if not file.filename:
            raise ValueError("没有上传文件")

        # 保存文件到 data 目录
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            if not content:
                raise ValueError("上传的文件为空")
            f.write(content)
        logger.info(f"文件已保存到 {file_path}")

        # 处理文件
        logger.info("正在处理文件内容...")

        # 调用文件处理函数
        text = process_file(file_path)

        # 根据参数选择合适处理函数
        processing_function = get_processing_function(document_type)

        # 调用文本处理函数
        logger.info("正在构建结构化结果...")
        structured_result = processing_function(text)

        logger.info(f"结构化结果: {structured_result}")

        return {"status": "success", "data": structured_result}
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
