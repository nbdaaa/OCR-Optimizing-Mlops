from fastapi import FastAPI

app = FastAPI(title="OCR-MLOps API")


@app.get("/health")
def health():
    return {"status": "ok"}
