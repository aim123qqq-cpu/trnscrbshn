# Personal Meeting Transcriber

Личный локальный сервис для видео и аудио встреч:

- загружает файл через браузер;
- извлекает аудио через `ffmpeg`, если он установлен;
- распознает речь локально через `faster-whisper` или через OpenAI Audio API;
- собирает протокол встречи через OpenAI Responses API;
- сохраняет `transcript.txt`, `meeting_report.md` и `meeting_report.docx`.

Публичная страница проекта на GitHub Pages: `https://aim123qqq-cpu.github.io/trnscrbshn/`

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

Для локального режима сервис использует системный `ffmpeg` или бинарь из `imageio-ffmpeg`.
Для API-режима нужны активные OpenAI API credits/billing. Если OpenAI-сводка недоступна, сервис все равно сохранит транскрипт и fallback-протокол.

## Full online deploy на Render

В репозитории есть `Dockerfile` и `render.yaml`.

1. Откройте Render Dashboard и создайте Blueprint из репозитория `aim123qqq-cpu/trnscrbshn`.
2. Укажите secret `OPENAI_API_KEY`.
3. После деплоя откройте URL Render-сервиса.

Для онлайн-режима по умолчанию выбран `OpenAI Audio API`, потому что локальный `faster-whisper` на облачном CPU может быть медленным и тяжелым.
