-- Минимальная схема БД для каталога китов + pgvector + тестовые записи

create extension if not exists vector;

-- ============================================================
-- 1. КИТЫ
-- ============================================================
create table if not exists whales (
    id bigserial primary key,
    whale_uid text not null unique,
    display_name text null,
    notes text null,
    created_at timestamp not null default now(),
    updated_at timestamp not null default now()
);

-- ============================================================
-- 2. ИЗОБРАЖЕНИЯ КАТАЛОГА
-- Храним путь к файлу, а не сам файл
-- ============================================================
create table if not exists catalog_images (
    id bigserial primary key,
    whale_id bigint not null references whales(id) on delete cascade,
    image_path text not null unique,
    width int null,
    height int null,
    created_at timestamp not null default now()
);

-- ============================================================
-- 3. ЭМБЕДДИНГИ
-- Отдельно от изображений, чтобы можно было хранить версии модели
-- ============================================================
create table if not exists image_embeddings (
    id bigserial primary key,
    catalog_image_id bigint not null references catalog_images(id) on delete cascade,
    model_name text not null,
    model_version text not null,
    embedding vector(864) not null,
    created_at timestamp not null default now(),
    unique (catalog_image_id, model_name, model_version)
);

-- ============================================================
-- 4. ДОПОЛНИТЕЛЬНЫЕ / ТЕСТОВЫЕ ЗАПИСИ ПО КИТУ
-- Минимальная таблица для формы на фронте
-- ============================================================
create table if not exists whale_observations (
    id bigserial primary key,
    whale_id bigint not null references whales(id) on delete restrict,
    observation_date date not null,
    location text null,
    comment_text text null,
    created_at timestamp not null default now()
);

-- ============================================================
-- ИНДЕКСЫ
-- ============================================================
create index if not exists idx_catalog_images_whale_id
    on catalog_images(whale_id);

create index if not exists idx_whale_observations_whale_id
    on whale_observations(whale_id);

-- Для ArcFace + L2-нормализации лучше начать с cosine distance
create index if not exists idx_image_embeddings_vector_cos
    on image_embeddings
    using hnsw (embedding vector_cosine_ops);