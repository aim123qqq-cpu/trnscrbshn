from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import cgi
except ImportError:  # pragma: no cover
    cgi = None


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work" / "jobs"
OUTPUT_DIR = ROOT / "outputs" / "jobs"
ENV_FILES = [ROOT / ".env.local", ROOT / ".env"]
SUPPORTED_OPENAI_AUDIO = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}


@dataclass
class Job:
    id: str
    filename: str
    backend: str
    language: str
    status: str = "queued"
    progress: str = "Waiting to start"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def load_dotenv() -> None:
    for env_file in ENV_FILES:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def update_job(job_id: str, **changes: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in changes.items():
            setattr(job, key, value)


def safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name).strip()
    return cleaned or "upload.bin"


def extract_audio(input_path: Path, job_dir: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    output_path = job_dir / "audio.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def transcribe_local(audio_path: Path, language: str) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc

    model_name = os.environ.get("WHISPER_MODEL", "small")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    lang = language or None
    segments, _info = model.transcribe(str(audio_path), language=lang, vad_filter=True)
    lines: list[str] = []
    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        if text:
            lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


def transcribe_openai(file_path: Path, language: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Check .env.local.")

    model = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    fields = {
        "model": model,
        "response_format": "json",
    }
    if language:
        fields["language"] = language

    response = multipart_post(
        "https://api.openai.com/v1/audio/transcriptions",
        api_key,
        fields,
        "file",
        file_path,
    )
    data = json.loads(response)
    return data.get("text", "").strip()


def multipart_post(url: str, api_key: str, fields: dict[str, str], file_field: str, file_path: Path) -> str:
    boundary = f"----codex-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    return read_url(request)


def summarize_transcript(transcript: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return fallback_report(transcript)

    model = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-5-mini")
    prompt = (
        "Собери протокол встречи на русском языке из транскрипта.\n"
        "Формат Markdown:\n"
        "# Протокол встречи\n"
        "## Краткое резюме\n"
        "## Договоренности\n"
        "## Задачи\n"
        "Таблица: Задача | Ответственный | Срок | Статус/контекст. Если данных нет, пиши \"не указано\".\n"
        "## Вопросы и риски\n"
        "## Полный транскрипт\n"
        "Не придумывай факты, имена, сроки и решения.\n\n"
        f"Транскрипт:\n{transcript}"
    )
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps({"model": model, "input": prompt}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    data = json.loads(read_url(request))
    text = data.get("output_text")
    if text:
        return text.strip()
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip() or fallback_report(transcript)


def read_url(request: urllib.request.Request) -> str:
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {details}") from exc


def fallback_report(transcript: str) -> str:
    return (
        "# Протокол встречи\n\n"
        "## Краткое резюме\n\n"
        "Сводка не создана: не найден `OPENAI_API_KEY` или API-запрос не выполнен.\n\n"
        "## Договоренности\n\n"
        "- не указано\n\n"
        "## Задачи\n\n"
        "| Задача | Ответственный | Срок | Статус/контекст |\n"
        "| --- | --- | --- | --- |\n"
        "| не указано | не указано | не указано | не указано |\n\n"
        "## Вопросы и риски\n\n"
        "- не указано\n\n"
        "## Полный транскрипт\n\n"
        f"{transcript}\n"
    )


def save_docx(markdown_text: str, output_path: Path) -> None:
    paragraphs = markdown_to_paragraphs(markdown_text)
    document_xml = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{html.escape(line)}</w:t></w:r></w:p>" for line in paragraphs
    )
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        docx.writestr("_rels/.rels", RELS_XML)
        docx.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS_XML)
        docx.writestr("word/document.xml", DOCUMENT_XML.format(body=document_xml))


def markdown_to_paragraphs(markdown_text: str) -> list[str]:
    lines: list[str] = []
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        lines.append(line)
    return lines or [""]


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_job(job_id: str, upload_path: Path) -> None:
    job_dir = upload_path.parent
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with JOBS_LOCK:
            job = JOBS[job_id]
            backend = job.backend
            language = job.language

        update_job(job_id, status="running", progress="Extracting audio")
        audio_path = extract_audio(upload_path, job_dir)

        if backend == "local" and not audio_path:
            raise RuntimeError("ffmpeg is required for local faster-whisper transcription.")

        update_job(job_id, progress="Transcribing")
        if backend == "local":
            transcript = transcribe_local(audio_path or upload_path, language)
        elif backend == "openai":
            target = audio_path if audio_path else upload_path
            if not audio_path and target.suffix.lower() not in SUPPORTED_OPENAI_AUDIO:
                raise RuntimeError("Install ffmpeg or upload a supported audio/video format for OpenAI transcription.")
            transcript = transcribe_openai(target, language)
        else:
            try:
                if not audio_path:
                    raise RuntimeError("ffmpeg is not available for local transcription.")
                transcript = transcribe_local(audio_path, language)
            except Exception:
                target = audio_path if audio_path else upload_path
                if not audio_path and target.suffix.lower() not in SUPPORTED_OPENAI_AUDIO:
                    raise
                transcript = transcribe_openai(target, language)

        if not transcript.strip():
            raise RuntimeError("Transcription completed but returned empty text.")

        transcript_path = output_dir / "transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")

        update_job(job_id, progress="Creating meeting report")
        report = summarize_transcript(transcript)
        report_path = output_dir / "meeting_report.md"
        report_path.write_text(report, encoding="utf-8")
        docx_path = output_dir / "meeting_report.docx"
        save_docx(report, docx_path)

        update_job(
            job_id,
            status="done",
            progress="Done",
            finished_at=time.time(),
            outputs={
                "transcript.txt": f"/download/{job_id}/transcript.txt",
                "meeting_report.md": f"/download/{job_id}/meeting_report.md",
                "meeting_report.docx": f"/download/{job_id}/meeting_report.docx",
            },
        )
    except Exception as exc:
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        update_job(job_id, status="error", progress="Failed", error=str(exc), finished_at=time.time())


class Handler(BaseHTTPRequestHandler):
    server_version = "PersonalTranscriber/0.1"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_html(INDEX_HTML)
        elif parsed.path.startswith("/status/"):
            self.handle_status(parsed.path.rsplit("/", 1)[-1])
        elif parsed.path.startswith("/download/"):
            self.handle_download(parsed.path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/upload":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if cgi is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "cgi module unavailable")
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        file_item = form["file"] if "file" in form else None
        if not file_item or not getattr(file_item, "filename", ""):
            self.send_error(HTTPStatus.BAD_REQUEST, "File is required")
            return

        job_id = uuid.uuid4().hex[:12]
        job_dir = WORK_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_filename(file_item.filename)
        upload_path = job_dir / filename
        with upload_path.open("wb") as target:
            shutil.copyfileobj(file_item.file, target)

        backend = form.getfirst("backend", "auto")
        if backend not in {"auto", "local", "openai"}:
            backend = "auto"
        language = form.getfirst("language", "ru").strip().lower()
        with JOBS_LOCK:
            JOBS[job_id] = Job(job_id, filename, backend, language)
        threading.Thread(target=process_job, args=(job_id, upload_path), daemon=True).start()
        self.send_json({"job_id": job_id})

    def handle_status(self, job_id: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = {
                "id": job.id,
                "filename": job.filename,
                "backend": job.backend,
                "status": job.status,
                "progress": job.progress,
                "error": job.error,
                "outputs": job.outputs,
                "elapsed_seconds": int((job.finished_at or time.time()) - job.created_at),
            }
        self.send_json(payload)

    def handle_download(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, job_id, filename = parts
        if filename not in {"transcript.txt", "meeting_report.md", "meeting_report.docx"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        file_path = OUTPUT_DIR / job_id / filename
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if filename.endswith(".txt"):
            content_type = "text/plain; charset=utf-8"
        elif filename.endswith(".md"):
            content_type = "text/markdown; charset=utf-8"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personal Transcriber</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111411;
      --surface: #181d19;
      --surface-2: #20281f;
      --panel: #f7fbf6;
      --panel-2: #eef5ed;
      --ink: #f4f8f3;
      --ink-panel: #162015;
      --muted: #aab7a8;
      --muted-panel: #64715f;
      --line: #313a31;
      --line-panel: #d8e2d4;
      --primary: #0d9488;
      --primary-strong: #0f766e;
      --accent: #ea580c;
      --danger: #dc2626;
      --success: #2f855a;
      --focus: #5eead4;
      --shadow: 0 24px 70px rgba(0, 0, 0, .36);
      --radius: 8px;
      --mono: "Fira Code", "Cascadia Mono", Consolas, monospace;
      --sans: "Fira Sans", "Segoe UI", Arial, sans-serif;
    }

    * { box-sizing: border-box; }
    html { min-width: 320px; }
    body {
      margin: 0;
      min-height: 100dvh;
      font-family: var(--sans);
      background: var(--bg);
      color: var(--ink);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(90deg, rgba(13, 148, 136, .08), transparent 34%),
        linear-gradient(180deg, rgba(234, 88, 12, .06), transparent 42%);
    }
    a, button, input, select { touch-action: manipulation; }
    button, input, select { font: inherit; }
    button:focus-visible, input:focus-visible, select:focus-visible, .file-control:focus-within {
      outline: 3px solid rgba(94, 234, 212, .78);
      outline-offset: 3px;
    }

    .app-shell {
      position: relative;
      min-height: 100dvh;
      display: grid;
      grid-template-columns: minmax(280px, 390px) minmax(0, 1fr);
    }
    .sidebar {
      padding: 32px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 32px;
      border-right: 1px solid var(--line);
      background: rgba(24, 29, 25, .92);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 44px;
    }
    .brand-mark {
      width: 44px;
      height: 44px;
      border: 1px solid rgba(94, 234, 212, .34);
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #0f1714;
      color: var(--focus);
      font-family: var(--mono);
      font-weight: 700;
    }
    .brand-text {
      display: grid;
      gap: 2px;
    }
    .brand-text strong { font-size: 15px; }
    .brand-text span { color: var(--muted); font-size: 13px; }
    h1 {
      margin: 0 0 18px;
      max-width: 10ch;
      font-size: clamp(40px, 6vw, 72px);
      line-height: .96;
      letter-spacing: 0;
    }
    .lead {
      max-width: 32rem;
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.6;
    }
    .system-list {
      display: grid;
      gap: 10px;
      margin: 36px 0 0;
      padding: 0;
      list-style: none;
    }
    .system-list li {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      color: var(--muted);
      line-height: 1.45;
    }
    .step-index {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: var(--surface-2);
      color: var(--focus);
      font: 700 13px var(--mono);
    }
    .sidebar-footer {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      min-height: 34px;
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      color: var(--muted);
      background: rgba(255, 255, 255, .03);
      font: 600 12px var(--mono);
    }

    .workspace {
      padding: 32px;
      display: grid;
      align-items: center;
    }
    .work-grid {
      width: min(100%, 980px);
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 18px;
      align-items: stretch;
    }
    .tool-card, .status-card {
      border: 1px solid var(--line-panel);
      border-radius: var(--radius);
      background: var(--panel);
      color: var(--ink-panel);
      box-shadow: var(--shadow);
    }
    .tool-card {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      border-bottom: 1px solid var(--line-panel);
      padding-bottom: 18px;
    }
    .eyebrow {
      margin: 0 0 7px;
      color: var(--primary-strong);
      font: 700 12px var(--mono);
      text-transform: uppercase;
    }
    h2 {
      margin: 0;
      font-size: clamp(22px, 3vw, 32px);
      line-height: 1.08;
      letter-spacing: 0;
    }
    .card-note {
      max-width: 22rem;
      margin: 8px 0 0;
      color: var(--muted-panel);
      line-height: 1.5;
    }
    .mode-badge {
      flex: 0 0 auto;
      min-height: 34px;
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 11px;
      background: #dff7f3;
      color: #0f766e;
      font: 700 12px var(--mono);
    }

    form { display: grid; gap: 18px; }
    label, .field-label {
      display: grid;
      gap: 8px;
      color: var(--ink-panel);
      font-weight: 700;
      font-size: 14px;
    }
    .hint {
      color: var(--muted-panel);
      font-weight: 500;
      font-size: 13px;
      line-height: 1.45;
    }
    .file-control {
      position: relative;
      min-height: 156px;
      display: grid;
      place-items: center;
      border: 1.5px dashed #a7b8a2;
      border-radius: var(--radius);
      background: var(--panel-2);
      text-align: center;
      transition: border-color .18s ease, background-color .18s ease, transform .18s ease;
    }
    .file-control:hover {
      border-color: var(--primary);
      background: #e7f5f1;
    }
    .file-control:active { transform: scale(.995); }
    .file-control input {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
      cursor: pointer;
    }
    .file-visual {
      display: grid;
      gap: 9px;
      justify-items: center;
      padding: 18px;
    }
    .file-symbol {
      width: 48px;
      height: 56px;
      border: 2px solid var(--primary);
      border-radius: 8px;
      position: relative;
      background: #f9fffb;
    }
    .file-symbol::before, .file-symbol::after {
      content: "";
      position: absolute;
      left: 11px;
      right: 11px;
      height: 2px;
      background: var(--primary);
    }
    .file-symbol::before { top: 20px; }
    .file-symbol::after { top: 31px; }
    .file-title { font-weight: 800; }
    .file-name {
      max-width: min(440px, 72vw);
      color: var(--muted-panel);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(120px, 180px);
      gap: 14px;
    }
    input[type=text], select {
      width: 100%;
      min-height: 48px;
      border: 1px solid var(--line-panel);
      border-radius: 8px;
      padding: 11px 12px;
      background: #fff;
      color: var(--ink-panel);
    }
    select { cursor: pointer; }
    .actions {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      padding: 13px 18px;
      background: var(--primary);
      color: #fff;
      font-weight: 800;
      cursor: pointer;
      transition: background-color .18s ease, transform .18s ease, opacity .18s ease;
    }
    button:hover { background: var(--primary-strong); }
    button:active { transform: translateY(1px); }
    button:disabled {
      opacity: .56;
      cursor: wait;
      transform: none;
    }
    .action-note {
      margin: 0;
      color: var(--muted-panel);
      font-size: 13px;
      line-height: 1.45;
    }

    .status-card {
      padding: 18px;
      display: grid;
      align-content: start;
      gap: 16px;
      background: #fbfdf9;
    }
    .status-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .status-label {
      margin: 0;
      color: var(--muted-panel);
      font: 700 12px var(--mono);
      text-transform: uppercase;
    }
    .status-dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      background: var(--primary);
      box-shadow: 0 0 0 5px rgba(13, 148, 136, .12);
    }
    .status-card[data-state="error"] .status-dot { background: var(--danger); box-shadow: 0 0 0 5px rgba(220, 38, 38, .12); }
    .status-card[data-state="done"] .status-dot { background: var(--success); box-shadow: 0 0 0 5px rgba(47, 133, 90, .12); }
    .status-title {
      margin: 0;
      font-size: 21px;
      line-height: 1.2;
    }
    .status-copy {
      margin: 0;
      color: var(--muted-panel);
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .progress-track {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dfe8dd;
    }
    .progress-bar {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: var(--primary);
      transition: width .24s ease;
    }
    .status-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric {
      min-height: 62px;
      border: 1px solid var(--line-panel);
      border-radius: 8px;
      padding: 10px;
      background: var(--panel-2);
    }
    .metric span {
      display: block;
      color: var(--muted-panel);
      font: 700 11px var(--mono);
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 5px;
      font: 700 18px var(--mono);
      color: var(--ink-panel);
    }
    .links {
      display: grid;
      gap: 8px;
    }
    .links a {
      min-height: 44px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      border: 1px solid var(--line-panel);
      color: var(--primary-strong);
      text-decoration: none;
      font-weight: 800;
    }
    .links a::after {
      content: "↓";
      font-family: var(--mono);
    }

    @media (max-width: 980px) {
      .app-shell { grid-template-columns: 1fr; }
      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .brand { margin-bottom: 28px; }
      h1 { max-width: 12ch; }
    }
    @media (max-width: 760px) {
      .sidebar, .workspace { padding: 22px; }
      .work-grid, .row { grid-template-columns: 1fr; }
      .card-header, .actions { align-items: stretch; flex-direction: column; }
      .mode-badge { width: fit-content; }
      .tool-card { padding: 18px; }
      .status-meta { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
      }
    }
  </style>
</head>
<body>
<main class="app-shell">
  <section class="sidebar" aria-labelledby="page-title">
    <div>
      <div class="brand" aria-label="Personal Transcriber">
        <div class="brand-mark" aria-hidden="true">PT</div>
        <div class="brand-text">
          <strong>Personal Transcriber</strong>
          <span>локальная рабочая станция</span>
        </div>
      </div>
      <h1 id="page-title">Личный транскрибатор встреч</h1>
      <p class="lead">Загрузите запись. Сервис сделает транскрипт, протокол, задачи, риски и файлы для передачи коллегам.</p>
      <ol class="system-list" aria-label="Этапы обработки">
        <li><span class="step-index">01</span><span>Принимает видео или аудио встречи.</span></li>
        <li><span class="step-index">02</span><span>Распознает речь локально или через OpenAI Audio API.</span></li>
        <li><span class="step-index">03</span><span>Собирает протокол в TXT, Markdown и DOCX.</span></li>
      </ol>
    </div>
    <div class="sidebar-footer" aria-label="Форматы результата">
      <span class="pill">TXT</span>
      <span class="pill">MD</span>
      <span class="pill">DOCX</span>
      <span class="pill">local/API</span>
    </div>
  </section>

  <section class="workspace">
    <div class="work-grid">
      <form class="tool-card" id="uploadForm">
        <div class="card-header">
          <div>
            <p class="eyebrow">Новая обработка</p>
            <h2>Запись встречи → готовый протокол</h2>
            <p class="card-note">Лучше всего подходят записи Телемоста, Zoom, Meet, Teams и обычные аудиофайлы.</p>
          </div>
          <span class="mode-badge">private local</span>
        </div>

        <div class="field-label">
          <span>Файл встречи</span>
          <span class="hint">MP4, WebM, M4A, WAV, MP3, OGG или FLAC. Большие файлы обрабатываются дольше.</span>
          <div class="file-control">
            <input id="fileInput" name="file" type="file" accept="audio/*,video/*,.m4a,.mp4,.webm,.wav,.mp3,.ogg,.flac" required aria-describedby="fileName">
            <div class="file-visual" aria-hidden="true">
              <span class="file-symbol"></span>
              <span class="file-title">Выберите файл или перетащите его сюда</span>
              <span class="file-name" id="fileName">Файл пока не выбран</span>
            </div>
          </div>
        </div>

        <div class="row">
          <label for="backendSelect">Распознавание
            <select id="backendSelect" name="backend">
              <option value="auto">Авто: локально, потом API</option>
              <option value="local">Только faster-whisper</option>
              <option value="openai">OpenAI Audio API</option>
            </select>
          </label>
          <label for="languageInput">Язык
            <input id="languageInput" name="language" type="text" value="ru" maxlength="8" autocomplete="off" inputmode="latin">
          </label>
        </div>

        <div class="actions">
          <button id="submitBtn" type="submit">Запустить обработку</button>
          <p class="action-note">Окно можно оставить открытым: статус обновляется автоматически.</p>
        </div>
      </form>

      <aside class="status-card" id="statusCard" data-state="idle" aria-live="polite" aria-label="Статус обработки">
        <div class="status-top">
          <p class="status-label">Статус</p>
          <span class="status-dot" aria-hidden="true"></span>
        </div>
        <div>
          <h2 class="status-title" id="statusTitle">Готов к загрузке</h2>
          <p class="status-copy" id="statusCopy">Выберите запись встречи и запустите обработку.</p>
        </div>
        <div class="progress-track" aria-hidden="true">
          <div class="progress-bar" id="progressBar"></div>
        </div>
        <div class="status-meta">
          <div class="metric"><span>Время</span><strong id="elapsedValue">0с</strong></div>
          <div class="metric"><span>Режим</span><strong id="backendValue">auto</strong></div>
        </div>
        <div class="links" id="downloadLinks" aria-label="Файлы результата"></div>
      </aside>
    </div>
  </section>
</main>
<script>
const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const fileName = document.querySelector("#fileName");
const statusCard = document.querySelector("#statusCard");
const statusTitle = document.querySelector("#statusTitle");
const statusCopy = document.querySelector("#statusCopy");
const progressBar = document.querySelector("#progressBar");
const elapsedValue = document.querySelector("#elapsedValue");
const backendValue = document.querySelector("#backendValue");
const linksBox = document.querySelector("#downloadLinks");
const button = document.querySelector("#submitBtn");
let timer = null;

const progressByStatus = {
  queued: 12,
  running: 58,
  done: 100,
  error: 100
};

const progressLabels = {
  "Waiting to start": "Ожидает запуска",
  "Extracting audio": "Извлекаю аудио",
  "Transcribing": "Распознаю речь",
  "Creating meeting report": "Собираю протокол",
  "Done": "Готово",
  "Failed": "Ошибка"
};

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files?.[0]?.name || "Файл пока не выбран";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = "Загружаю...";
  setStatus("running", "Загружаю файл", "Передаю запись локальному сервису.", 8, 0, form.backend.value);
  linksBox.innerHTML = "";

  try {
    const response = await fetch("/upload", { method: "POST", body: new FormData(form) });
    if (!response.ok) throw new Error("Сервер не принял файл. Проверьте формат и попробуйте снова.");
    const data = await response.json();
    poll(data.job_id);
  } catch (error) {
    setStatus("error", "Ошибка загрузки", error.message || "Не удалось загрузить файл.", 100, 0, form.backend.value);
    button.disabled = false;
    button.textContent = "Запустить обработку";
  }
});

async function poll(jobId) {
  clearTimeout(timer);
  try {
    const response = await fetch(`/status/${jobId}`);
    if (!response.ok) throw new Error("Не удалось получить статус задачи.");
    const job = await response.json();
    const title = progressLabels[job.progress] || job.progress || "Обработка";
    const copy = job.status === "done"
      ? "Файлы готовы. Скачайте нужный формат ниже."
      : job.status === "error"
        ? (job.error || "Обработка остановилась с ошибкой.")
        : `Задача ${job.id} выполняется.`;
    setStatus(job.status, title, copy, progressByStatus[job.status] || 30, job.elapsed_seconds, job.backend);

    if (job.status === "done") {
      linksBox.innerHTML = Object.entries(job.outputs)
        .map(([name, href]) => `<a href="${href}">${escapeHTML(name)}</a>`)
        .join("");
      button.disabled = false;
      button.textContent = "Запустить обработку";
      return;
    }
    if (job.status === "error") {
      button.disabled = false;
      button.textContent = "Запустить обработку";
      return;
    }
    timer = setTimeout(() => poll(jobId), 1500);
  } catch (error) {
    setStatus("error", "Потеряна связь", error.message || "Статус временно недоступен.", 100, 0, form.backend.value);
    button.disabled = false;
    button.textContent = "Запустить обработку";
  }
}

function setStatus(state, title, copy, progress, elapsed, backend) {
  statusCard.dataset.state = state;
  statusTitle.textContent = title;
  statusCopy.textContent = copy;
  progressBar.style.width = `${progress}%`;
  elapsedValue.textContent = `${elapsed || 0}с`;
  backendValue.textContent = backend || "auto";
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[char]);
}
</script>
</body>
</html>"""


CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOCUMENT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""

DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}<w:sectPr/></w:body>
</w:document>"""


def main() -> None:
    load_dotenv()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Personal Transcriber is running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
