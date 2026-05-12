# WhaleDetector — Frontend (Angular + Ionic)

Фронтенд для системы идентификации гренландских китов Охотского моря.

## Стек технологий
- **Angular 16** — основной фреймворк
- **Ionic 7** — UI компоненты и мобильный стиль
- **RxJS** — реактивное управление состоянием
- **SCSS** — стили с ocean-темой

## Структура страниц

| Страница | Путь | Описание |
|----------|------|----------|
| Загрузка видео | `/home` | Drag & drop видео, прогресс загрузки/обработки |
| Выбор кадров | `/frames` | Grid лучших кадров, выбор для идентификации |
| Результаты | `/results` | Совпадения с каталогом, одобрение/отклонение |

## Поток работы

```
Загрузка видео → [бэкенд: детекция + извлечение кадров]
      ↓
Отображение лучших кадров (grid) → Выбор оператором
      ↓
[бэкенд: Reid идентификация]
      ↓
Результаты (топ-5 совпадений или "Новый кит") → Одобрить / Отклонить
```

## Установка

```bash
cd frontend
npm install
```

## Запуск в разработке

Убедитесь, что бэкенд запущен на `http://localhost:8000`.

```bash
npm start
```

Прокси-конфиг автоматически перенаправляет `/api` → `http://localhost:8000`.

## Production сборка

```bash
npm run build:prod
```

Статика окажется в папке `www/`.

## API эндпоинты (ожидаемые от бэкенда)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/upload` | Загрузка видео (multipart/form-data: `video`) → `{ job_id }` |
| `GET` | `/api/job/{id}/status` | Статус обработки → `{ status, progress, message, stage }` |
| `GET` | `/api/job/{id}/frames` | Лучшие кадры → `{ frames: Frame[] }` |
| `POST` | `/api/identify` | Идентификация → `{ job_id, frame_ids }` → `IdentifyResponse` |

### Типы данных

```typescript
// Статус задания
{ status: 'pending'|'processing'|'completed'|'failed', progress?: number, message?: string, stage?: string }

// Кадр
{ id: string, url: string, score: number, timestamp?: number }

// Результат идентификации
{
  results: [{
    frame_id?: string,
    frame_url?: string,
    matches: [{ whale_id: string, confidence: number, thumbnail_url?: string }],
    is_new: boolean
  }]
}
```

## Конфигурация бэкенда

Измените `src/environments/environment.ts`:

```typescript
export const environment = {
  production: false,
  apiUrl: 'http://your-backend-url/api'  // для production без прокси
};
```

Или измените `proxy.conf.json` для изменения порта бэкенда.

