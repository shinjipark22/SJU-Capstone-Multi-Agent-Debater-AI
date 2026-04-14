import uvicorn
from fastapi import FastAPI

from .routes import router

app = FastAPI(
    title="Debate Evaluation API",
    description="토론 전후 사용자 근거를 평가하는 API",
    version="1.0.0",
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
