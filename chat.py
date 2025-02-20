# -*- coding: utf-8 -*-

from langchain_ollama import ChatOllama
from logger_config import logger
import json

# 构造 LLM
model = ChatOllama(format="json", model="qwen2.5:7b", base_url="http://127.0.0.1:11434")

# 定义系统消息
system_message = "回答用户查询。请确保返回的数据包含所有相关信息。用中文回答.用json格式输出，不要包含json标签，并以纯JSON格式返回。"


# 方法：接收前端的消息，并返回结构化数据
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

# 新增的chat_with_frontend方法：处理前端请求
def chat_with_model(message: str) -> dict:
    """
    接收前端消息并返回大模型的回应。
    :param message: 前端发送的消息
    :return: 大模型的结构化回复
    """
    try:
        logger.info(f"收到前端消息: {message}")

        # 调用 process_text 方法，获取模型的结构化回应
        model_response = process_text(message)

        logger.info(f"大模型回应: {model_response}")

        # 返回模型的回应
        return {"status": "success", "message": model_response}

    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
