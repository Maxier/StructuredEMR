# -*- coding: utf-8 -*-

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json
from logger_config import logger

# 构造 LLM
model = ChatOllama(format="json", model="qwen2.5:7b", base_url="http://127.0.0.1:11434")

# 定义系统消息
system_message = "回答用户查询。请确保返回的数据包含所有相关信息。用中文回答.用json格式输出，不要包含json标签，并以纯JSON格式返回。"


# 方法：接收文本并返回结构化数据


def process_text(query: str) -> dict:
    try:
        # 构造输入 Prompt
        logger.info(f"query的内容: {query}")
        input_prompt = f"{system_message}\n\n用户问题：{query}"

        # 调用模型获取响应
        raw_output = model.invoke(input=input_prompt)

        # 提取模型输出内容（确保是字符串）
        if hasattr(raw_output, "content"):
            response_content = raw_output.content
        else:
            raise ValueError("Model output does not contain a valid 'content' attribute.")

        # 确保返回的数据是有效的 JSON
        logger.info(f"response_content: {response_content}")

        # 返回结构化数据
        return response_content

    except json.JSONDecodeError as e:
        # 如果返回的数据不是有效的 JSON，返回错误信息
        return {"status": "error", "message": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        # 处理其他异常，返回错误信息
        return {"status": "error", "message": str(e)}
