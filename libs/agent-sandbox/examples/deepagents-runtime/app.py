"""Reference HTTP runtime for agent-sandbox file and command operations."""

from __future__ import annotations

import os
import shlex
import subprocess
import urllib.parse

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.realpath(os.environ.get("SANDBOX_RUNTIME_DIR", "/workspace"))


class ExecuteRequest(BaseModel):
    """Request model for /execute."""

    command: str


class ExecuteResponse(BaseModel):
    """Response model for /execute."""

    stdout: str
    stderr: str
    exit_code: int


def safe_path(file_path: str) -> str:
    """Resolve a runtime path and refuse traversal outside BASE_DIR."""
    full_path = os.path.realpath(os.path.join(BASE_DIR, file_path.lstrip("/")))
    if os.path.commonpath([BASE_DIR, full_path]) != BASE_DIR:
        msg = f"Path must remain within {BASE_DIR}"
        raise ValueError(msg)
    return full_path


app = FastAPI(title="DeepAgents Sandbox Runtime")


@app.get("/")
async def health_check() -> dict[str, str]:
    """Return runtime health."""
    return {"status": "ok", "base_dir": BASE_DIR}


@app.post("/execute", response_model=ExecuteResponse)
async def execute_command(request: ExecuteRequest) -> ExecuteResponse:
    """Execute a command with cwd fixed to BASE_DIR."""
    try:
        result = subprocess.run(
            shlex.split(request.command),
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
            check=False,
        )
    except Exception as error:
        return ExecuteResponse(stdout="", stderr=str(error), exit_code=1)
    return ExecuteResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.returncode,
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a file under BASE_DIR."""
    try:
        target = safe_path(file.filename or "")
    except ValueError:
        return JSONResponse(status_code=403, content={"message": "Access denied"})
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(await file.read())
    return JSONResponse(status_code=200, content={"message": "ok"})


@app.get("/download/{encoded_file_path:path}")
async def download_file(encoded_file_path: str) -> FileResponse | JSONResponse:
    """Download a file under BASE_DIR."""
    try:
        target = safe_path(urllib.parse.unquote(encoded_file_path))
    except ValueError:
        return JSONResponse(status_code=403, content={"message": "Access denied"})
    if not os.path.isfile(target):
        return JSONResponse(status_code=404, content={"message": "File not found"})
    return FileResponse(target, media_type="application/octet-stream")


@app.get("/list/{encoded_file_path:path}")
async def list_files(encoded_file_path: str) -> JSONResponse:
    """List directory entries under BASE_DIR."""
    try:
        target = safe_path(urllib.parse.unquote(encoded_file_path))
    except ValueError:
        return JSONResponse(status_code=403, content={"message": "Access denied"})
    if not os.path.isdir(target):
        return JSONResponse(status_code=404, content={"message": "Not a directory"})
    entries = []
    with os.scandir(target) as iterator:
        for entry in iterator:
            stat = entry.stat()
            entries.append(
                {
                    "name": entry.name,
                    "size": stat.st_size,
                    "type": "directory" if entry.is_dir() else "file",
                    "mod_time": stat.st_mtime,
                }
            )
    return JSONResponse(status_code=200, content=entries)


@app.get("/exists/{encoded_file_path:path}")
async def exists(encoded_file_path: str) -> JSONResponse:
    """Report whether a path exists under BASE_DIR."""
    try:
        target = safe_path(urllib.parse.unquote(encoded_file_path))
    except ValueError:
        return JSONResponse(status_code=403, content={"exists": False})
    return JSONResponse(status_code=200, content={"exists": os.path.exists(target)})
