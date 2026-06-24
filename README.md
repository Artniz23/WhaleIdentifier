# WhaleIdentifier

Программная система для идентификации индивидуальных особей гренландского кита 
на основе автоматического отбора информативных кадров 
и сопоставления визуальных признаков.

Основной функционал:
- выделение объектов (детекция и трекинг);
- отбор лучших кадров по трекам;
- ReID-сопоставление с каталогом известных особей;
- просмотр и подтверждение результатов в веб-интерфейсе.

Проект состоит из двух основных частей:
- `backend/` - API и ML-логика (FastAPI + YOLO + ReID + PostgreSQL/pgvector);
- `frontend/` - пользовательский интерфейс (Angular + Ionic).

## Содержание

- [1. Возможности проекта](#1-возможности-проекта)
- [2. Архитектура и поток данных](#2-архитектура-и-поток-данных)
- [3. Быстрый старт](#3-быстрый-старт)
- [4. Backend](#4-backend)
- [5. Frontend](#5-frontend)
- [6. API (основные endpoints)](#6-api-основные-endpoints)
- [7. Работа с базой данных и каталогом](#7-работа-с-базой-данных-и-каталогом)
- [8. Структура проекта](#8-структура-проекта)
- [9. Типовые проблемы и решения (Troubleshooting)](#9-типовые-проблемы-и-решения-troubleshooting)
- [10. Материалы ВКР](#10-материалы-вкр)

## 1. Возможности проекта

- Загрузка видео с фронтенда.
- Асинхронная обработка видео:
  - детекция и трекинг китов;
  - фильтрация и ранжирование кадров по качеству;
  - сохранение изображений.
- Выбор пользователем кадров в интерфейсе.
- Идентификация выбранных кадров по каталогу известных китов через эмбеддинги.
- Выдача топ-N совпадений и признака "новая особь".

## 2. Архитектура и поток данных

Общий сценарий работы:

1. Пользователь загружает видео через frontend.
2. Backend создает задачу и запускает обработку.
3. После завершения трекинга backend возвращает треки с кадрами.
4. Пользователь выбирает лучшие кадры по трекам.
5. Backend запускает задачу идентификации.
6. Результаты сопоставляются с каталогом в PostgreSQL + pgvector.

Упрощенная схема:

```text
Frontend (Angular/Ionic)
  -> POST /api/upload
  -> GET /api/job/{job_id}/status
  -> GET /api/job/{job_id}/tracks
  -> POST /api/identify
  -> GET /api/identify/{identify_job_id}/status

Backend (FastAPI)
  -> YOLO трекинг
  -> ReID эмбеддинги/классификация
  -> PostgreSQL + pgvector (каталог)
```

## 3. Быстрый старт

Ниже минимальная последовательность для локального запуска (рекомендуется в WSL/Linux).

1) Поднять backend.
2) Поднять frontend.
3) Открыть UI и протестировать загрузку видео.

> Важно: в `frontend/proxy.conf.json` backend ожидается на `http://localhost:8000`.
> Поэтому backend удобно запускать на порту `8000`.

## 4. Backend

### 4.1 Технологии

- Python + FastAPI + Uvicorn
- PyTorch, Ultralytics (YOLO)
- PostgreSQL + pgvector

### 4.2 Требования

- Python 3.10+ (на практике часто используют 3.12)
- `pip`
- PostgreSQL 14+ (желательно) с расширением `pgvector`
- Наличие весов моделей в каталоге `backend/models/`

### 4.3 Установка зависимостей

```bash
cd WhaleIdentifier/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.4 Настройка окружения

Создайте `.env` на основе примера:

```bash
cd WhaleIdentifier/backend
cp .env.example .env
```

Переменные в `backend/.env.example`:

- `REID_CKPT_PATH` - путь к checkpoint ReID-модели.
- `YOLO_CKPT_PATH` - путь к checkpoint детектора.
- `YOLO_PRETRAINED_WEIGHTS_PATH` - путь к pretrain-весам (если используется).
- `CATALOG_IMAGE_DIR` - папка с изображениями каталога.
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` - параметры подключения к PostgreSQL.

Пример заполнения:

```dotenv
REID_CKPT_PATH=best.ckpt
YOLO_CKPT_PATH=best.pt
YOLO_PRETRAINED_WEIGHTS_PATH=yolo11s-obb.pt

CATALOG_IMAGE_DIR=/path/to/catalog_images

DB_HOST=localhost
DB_PORT=5432
DB_NAME=whale_reid_db
DB_USER=whale_app
DB_PASSWORD=your_password
```

### 4.5 Запуск backend

Рекомендуемый вариант (совместим с прокси фронтенда):

```bash
cd WhaleIdentifier/backend
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Альтернатива через `python server.py` (в коде указан порт `8005`):

```bash
cd WhaleIdentifier/backend
source .venv/bin/activate
python server.py
```

### 4.6 Что важно проверить перед запуском

- Существуют файлы весов:
  - `backend/models/detection/obb/weights/best.pt`
  - `backend/models/reid/b7/0/best.ckpt`
- PostgreSQL доступен, БД создана, расширение `vector` установлено.
- Активировано правильное виртуальное окружение.

## 5. Frontend

### 5.1 Технологии

- Angular 16
- Ionic 7
- TypeScript + SCSS

### 5.2 Установка зависимостей

```bash
cd WhaleIdentifier/frontend
npm install
```

### 5.3 Запуск в режиме разработки

```bash
cd WhaleIdentifier/frontend
npm start
```

Команда использует `ng serve --proxy-config proxy.conf.json --open`.

Проксируются пути:
- `/api` -> `http://localhost:8000`
- `/static` -> `http://localhost:8000`

### 5.4 Production-сборка

```bash
cd WhaleIdentifier/frontend
npm run build:prod
```

Также доступны скрипты:

```bash
cd WhaleIdentifier/frontend
npm run build
npm run electron
npm run electron-build
```

## 6. API (основные endpoints)

Ниже ключевые маршруты из `backend/server.py`.

### 6.1 Загрузка видео

- `POST /api/upload`
- Формат: `multipart/form-data`, поле `video`
- Ответ: `{ "job_id": "job_xxx" }`

### 6.2 Статус обработки видео

- `GET /api/job/{job_id}/status`
- Ответ содержит: `status`, `progress`, `stage`, `tracks_count`, `message`

### 6.3 Получение треков

- `GET /api/job/{job_id}/tracks`
- Ответ: `{ "tracks": [...] }`

### 6.4 Запуск идентификации

- `POST /api/identify`
- Тело запроса:

```json
{
  "job_id": "job_xxx",
  "selections": [
	{
	  "track_id": 1,
	  "frame_ids": ["job_xxx_track_1_1", "job_xxx_track_1_2"]
	}
  ]
}
```

### 6.5 Статус идентификации

- `GET /api/identify/{identify_job_id}/status`
- В статусе `completed` возвращаются результаты сопоставления.

## 7. Работа с базой данных и каталогом

### 7.1 Создание схемы БД

SQL-схема расположена в `backend/catalog/sql/create_database.sql`.

Пример применения:

```bash
psql -h localhost -U whale_app -d whale_reid_db -f WhaleIdentifier/backend/catalog/sql/create_database.sql
```

### 7.2 Импорт каталога китов в PostgreSQL

Скрипт: `backend/catalog/create_catalog.py`

```bash
cd WhaleIdentifier/backend
source .venv/bin/activate
python catalog/create_catalog.py
```

Перед запуском убедитесь, что заполнены переменные в `.env`:
- `REID_CKPT_PATH`
- `CATALOG_IMAGE_DIR`
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

## 8. Структура проекта

```text
WhaleIdentifier/
  backend/
	server.py                 # FastAPI приложение и основные endpoints
	requirements.txt          # Python зависимости
	catalog/                  # Импорт каталога и SQL-скрипты
	services/                 # Логика detection/reid/catalog
	models/                   # Веса и чекпоинты моделей
	dataset/                  # Скрипты подготовки датасетов
	results/, uploads/, videos/

  frontend/
	src/app/                  # Компоненты, страницы, сервисы
	src/environments/         # Конфигурация apiUrl
	proxy.conf.json           # Dev-прокси до backend
	package.json              # npm-скрипты и зависимости
```

## 9. Типовые проблемы и решения (Troubleshooting)

### Проблема 1: фронтенд не видит backend

- Симптом: ошибки `404/502` при запросах `/api/...`.
- Проверьте, что backend запущен на порту `8000` (или измените `frontend/proxy.conf.json`).

### Проблема 2: ошибка загрузки модели

- Симптом: `FileNotFoundError` для `.pt` или `.ckpt`.
- Проверьте пути к весам и наличие файлов в `backend/models/`.

### Проблема 3: ошибка подключения к БД

- Симптом: `psycopg2.OperationalError`.
- Проверьте доступность PostgreSQL и параметры `DB_*`.
- Убедитесь, что установлено расширение `vector`.

### Проблема 4: долгий запуск или нехватка памяти

- Уменьшите нагрузку (например, тестируйте на коротком видео).
- Используйте CPU-режим при отсутствии стабильной CUDA-конфигурации.

## 10. Материалы ВКР

- Презентация: [Презентация](https://disk.yandex.ru/d/xkQrTDfBPeDV2w)
- Отчет ВКР: [Отчет ВКР](https://disk.yandex.ru/i/RBsnpWNUBL0hLg)
- Демо Видео: [Демо Видео](https://disk.yandex.ru/i/M_f-ogMZ_gpopQ)

---
