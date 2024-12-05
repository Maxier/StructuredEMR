
from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from upload_file import router as upload_file_router
from logger_config import logger

# 初始化 FastAPI 应用
app = FastAPI()  # 2. 创建一个 FastAPI 实例

origins = [
    "http://localhost:8080",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 将子路由挂载到主应用
app.include_router(upload_file_router, prefix="/api", tags=["上传文件"])


