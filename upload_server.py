from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "/tmp/bib_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in [".mp4", ".avi", ".mkv"]:
            return JSONResponse(
                {"status": "error", "message": f"Format {ext} tidak didukung."},
                status_code=400
            )

        unique_name = f"{uuid.uuid4()}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)

        async with aiofiles.open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB per chunk
                await f.write(chunk)

        return JSONResponse({
            "status": "ok",
            "filename": file.filename,
            "saved_as": unique_name,
            "path": save_path
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


class DeleteRequest(BaseModel):
    path: str


@app.delete("/upload/video")
async def delete_video(body: DeleteRequest):
    try:
        path = body.path

        # Keamanan: hanya boleh hapus file di UPLOAD_DIR
        if not path.startswith(UPLOAD_DIR):
            return JSONResponse(
                {"status": "error", "message": "Path tidak diizinkan."},
                status_code=403
            )

        if os.path.exists(path):
            os.remove(path)
            return JSONResponse({"status": "ok", "message": "File dihapus."})
        else:
            return JSONResponse({"status": "ok", "message": "File tidak ditemukan, skip."})

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )