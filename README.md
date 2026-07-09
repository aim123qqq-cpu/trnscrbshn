# Personal Meeting Transcriber

Личный локальный сервис для видео и аудио встреч:

- загружает файл через браузер;
- извлекает аудио через `ffmpeg`, если он установлен;
- распознает речь локально через `faster-whisper` или через OpenAI Audio API;
- собирает протокол встречи через OpenAI Responses API;
- сохраняет `transcript.txt`, `meeting_report.md` и `meeting_report.docx`.

## Быстрый старт

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Откройте `http://127.0.0.1:8765`.

## Настройки

Созданный ключ уже сохранен в `.env.local` как `OPENAI_API_KEY`.

Переменные:

- `OPENAI_API_KEY` - ключ для сводки и API-транскрибации.
- `OPENAI_SUMMARY_MODEL` - модель для протокола, по умолчанию `gpt-5-mini`.
- `OPENAI_TRANSCRIBE_MODEL` - модель API-транскрибации, по умолчанию `gpt-4o-mini-transcribe`.
- `WHISPER_MODEL` - локальная модель faster-whisper, по умолчанию `small`.
- `TRANSCRIBE_BACKEND` - `local`, `openai` или `auto`, по умолчанию `auto`.

Для локального режима нужен установленный `ffmpeg`. Для API-режима `ffmpeg` желателен, но не обязателен для `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`, `ogg`, `flac`.
