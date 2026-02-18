# FaceID Baseline

Стартовый этап проекта: обучение модели идентификации лиц на основе эмбеддингов FaceNet и классификатора.

## 1. Структура датасета

```text
dataset/
  person_001/
    img1.jpg
    img2.jpg
  person_002/
    img1.jpg
```

Рекомендуется минимум 10-20 фото на каждого человека в разных условиях.

## 2. Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Веса FaceNet теперь подтягиваются автоматически при первом запуске.
Если нужно скачать заранее:

```bash
python download_weights.py
```

## 3. Обучение

```bash
python train.py --dataset-dir dataset --output-dir artifacts
```

Артефакты после обучения:
- `artifacts/faceid_classifier.joblib` - классификатор и label encoder
- `artifacts/metrics.json` - метрики
- `artifacts/embeddings.npy` - эмбеддинги обучающей выборки
- `artifacts/faceid_identities.db` - SQLite с центроидами эмбеддингов по людям

Важно: в артефактах не хранятся исходные изображения, только биометрические векторы.

## 4. Проверка на одном изображении

```bash
python predict.py --model-path artifacts/faceid_classifier.joblib --image-path test.jpg
```

## 5. Что делать дальше

1. Добавить порог верификации (`unknown` при низкой уверенности).
2. Перейти на inference-режим для Raspberry Pi (квантованная модель / TFLite).
3. Подключить поток с USB-камеры и управление реле.
4. Добавить логирование инцидентов и Telegram-уведомления.

## 6. Камера: регистрация и проверка

Регистрация пользователя через камеру:

```bash
python camera_app.py register --name artem --samples 20 --db-path artifacts/faceid_identities.db
```

Проверка в реальном времени:

```bash
python camera_app.py verify --db-path artifacts/faceid_identities.db --threshold 0.70 --show-score
```

Журнал событий:
- события доступа пишутся в таблицу `access_events` (SQLite);
- хранение временное, по умолчанию `7` дней (`--event-ttl-days 7`);
- одинаковые события пишутся с интервалом по умолчанию `1.2` сек (`--event-cooldown-sec`).
- при необходимости включаются Telegram-оповещения (`--telegram-enabled`) по сериям отказов.

Управление:
- `q` - выйти из окна камеры.

Примечание: регистрация сохраняет только усредненный эмбеддинг лица в SQLite, без исходных фото.

## 7. GUI (окно с кнопками)

Запуск окна:

```bash
python gui_app.py
```

В окне:
- при первом запуске нажмите `Download Weights`;
- введите имя пользователя и нажмите `Start Registration`;
- для распознавания нажмите `Start Verification`;
- в блоке `Verification` можно менять `liveness mode`, порог и anti-spoof параметры;
- `Стоп` завершает текущий процесс.
