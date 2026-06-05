from fastapi import FastAPI

from src.api.routers import data, training, models, deploy, cicd

app = FastAPI(title="OCR-MLOps API")

app.include_router(data.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(deploy.router)
app.include_router(cicd.router)


@app.get("/health")
def health():
    return {"status": "ok"}
