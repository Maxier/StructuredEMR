# -*- coding: utf-8 -*-
import json
import logging

from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
from pydantic import BaseModel, Field

# 构造模型
model = OllamaLLM(format="json", model="qwen2.5:7b", base_url="http://127.0.0.1:11434")


# 定义数据结构
class Person(BaseModel):
    姓名: Any = Field(default="", description="姓名")
    性别: Any = Field(default="", description="性别")
    年龄: Any = Field(default="", description="年龄")
    入院日期: Any = Field(default="", description="入院日期")
    主诉: Any = Field(default="", description="主诉")
    现病史: Any = Field(default="", description="现病史")
    既往史: Any = Field(default="", description="既往史")
    其他信息: Dict[str, Any] = Field(default_factory=dict, description="其他未定义的字段")


# 设置解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 设置提示模板
system_message = "回答用户查询。用JSON格式输出，不要包含多余文本，仅返回有效JSON。"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 构造链式调用
chain = LLMChain(llm=model, prompt=prompt)


# 修复后的 split_fields 和 process_medical_text 函数

def split_fields(data: dict, model: BaseModel) -> dict:
    """
    根据模型字段分离结构化数据和额外数据。
    """
    allowed_fields = model.__fields__.keys()
    structured_data = {key: data.get(key, "") for key in allowed_fields if key != "其他信息"}  # 排除其他信息
    extra_data = {key: value for key, value in data.items() if key not in allowed_fields}
    return {"structured": structured_data, "extra": extra_data}


def process_medical_text(text: str) -> dict:
    """
    处理输入的医疗文本并返回结构化数据。
    :param text: 输入的医疗文本
    :return: 结构化数据字典
    """
    try:
        logging.info(f"接收到的查询文本: {text}")  # 打印输入文本
        raw_output = chain.invoke({"query": text})
        logging.info(f"模型返回的完整结果: {raw_output}")  # 打印模型的完整返回值

        # 修正 raw_content 的提取逻辑
        if isinstance(raw_output, dict):
            raw_content = raw_output.get("text", "")
        elif isinstance(raw_output, str):
            raw_content = raw_output
        else:
            raise ValueError(f"返回的类型不是字符串或字典，当前类型: {type(raw_output)}")

        logging.info(f"提取的文本内容: {raw_content}")  # 打印从字典提取的内容

        raw_content = raw_content.strip()  # 去掉多余空格或换行符
        parsed_data = json.loads(raw_content)  # 解析 JSON 数据

        # 分离字段
        split_data = split_fields(parsed_data, Person)
        structured_data = split_data["structured"]
        extra_data = split_data["extra"]

        # 填充模型字段，确保其他信息不会重复
        person = Person(**structured_data, 其他信息=extra_data)
        logging.info(f"最终结构化结果: {person.dict()}")  # 打印结构化结果
        return person.dict()

    except json.JSONDecodeError as e:
        logging.info(f"JSON 解析错误，返回的内容: {raw_content}，错误信息: {str(e)}")
        return {"status": "error", "message": f"无效的 JSON 返回值: {raw_content}，错误信息: {str(e)}"}
    except Exception as e:
        logging.info(f"处理过程中发生错误: {str(e)}")
        return {"status": "error", "message": str(e)}
