# Извлечение задач из транскриптов встреч

Проект решает тестовое задание: из русскоязычных Zoom-транскриптов нужно
получить итоговый `pandas.DataFrame` с задачами, статусами, ответственными,
сроками и проверяемым обоснованием из исходного текста.

Главная идея проекта: не просить LLM "прочитать всё и придумать таблицу", а
держать модель в контролируемом контуре. Код сам находит релевантные фрагменты,
режет транскрипт на anchors, проверяет evidence, нормализует даты, запускает
несколько прогонов и собирает финальный результат только через consensus.

Подробный отчёт с метриками, сравнением baseline и дальнейшим планом:
[report.md](report.md).

## Текущий статус

На последнем сохранённом прогоне `tasks_verified.sqlite`:

| Метрика | Значение |
| --- | ---: |
| Транскриптов в проверке | 3 |
| LLM-прогонов на транскрипт | 5 |
| Итоговых строк в DataFrame | 23 |
| Precision против golden-разметки | 91.3% |
| Recall против golden-разметки | 63.6% |
| Строгая воспроизводимость 5/5 по consensus-группам | 57.5% |
| Средний минимальный pairwise Jaccard | 65.5% |
| Файлов, прошедших порог стабильности 90% | 0 из 3 |
| Unit-тесты | 90 passed |

Вывод честный: контроль галлюцинаций уже неплохой, но целевая
воспроизводимость `90%` пока не достигнута. Основная проблема теперь не в
формате DataFrame и не в SQLite, а в том, что LLM всё ещё по-разному
формулирует и выбирает задачи между пятью прогонами.

Лучший строгий прогон без verifier (`tasks_anchored.sqlite`) дал более сильную
стабильность: `65.7%` strict 5/5 и `76.5%` средний минимальный Jaccard. Поэтому
verifier сейчас рассматривается как экспериментальный слой, а не как финальная
победа.

## Итоговый DataFrame

Сам `DataFrame` соответствует формату из задания и содержит только 5 колонок:

```text
Блок | Задача | Ответственный | Срок | Обоснование
```

Служебные поля `support_count`, `support_ratio`, `run_indices`,
`verification_status` не попадают в `DataFrame`. Они хранятся отдельно в
`StabilityReport` и SQLite, чтобы можно было аудировать стабильность, не ломая
формат таблицы из задания.

Блоки:

- `Выполненные`
- `Невыполненные`
- `Новые`

Для `Новые` срок обязателен и должен быть подтверждён в evidence. Для
`Выполненные` срок очищается, потому что дата рядом с выполненной фразой часто
является соседним дедлайном, а не сроком выполненной задачи.

## Быстрый запуск

```bash
cd /Users/niksol/PycharmProjects/transcribtion
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` нужен ключ:

```env
OPENAI_API_KEY=sk-...
```

Запуск всех трёх транскриптов:

```bash
python main.py --all
```

По умолчанию это:

- 5 LLM-прогонов;
- strategy `anchored`;
- порог consensus `0.90`;
- near-consensus verifier включён;
- финальный `DataFrame` строится только из строк с поддержкой `5/5` и
  проверенных verifier-кандидатов.

Сохранить результат в SQLite:

```bash
python main.py --all --save-sqlite --sqlite-path tasks_verified.sqlite
```

Сравнить с golden-разметкой:

```bash
python main.py --all --eval-golden
```

Запустить аудит словарей без LLM:

```bash
python main.py --all --audit-lexicon
```

## Архитектура

```text
transcript.txt
-> parse_transcript
-> build_task_anchors
-> anchored LLM extraction
-> strict JSON Schema
-> postprocess validation
-> date normalization
-> dedupe
-> 5-run stability check
-> 90% consensus + optional verifier
-> pandas.DataFrame
-> optional SQLite
```

Что делает код до LLM:

- парсит Zoom-транскрипт в реплики;
- строит high-recall anchors вокруг сроков, поручений, статусов и финальных
  recap-блоков;
- сокращает входной текст перед API;
- объединяет пересекающиеся anchors;
- включает final-tail fallback, если coverage подозрительно низкий.

Что делает LLM:

- решает, есть ли в anchor задача;
- классифицирует блок;
- формулирует задачу;
- возвращает ответственного, сырой срок, evidence и `anchor_ids`;
- для near-consensus кандидатов отдельно проверяет, подтверждена ли строка
  evidence.

Что делает код после LLM:

- проверяет, что evidence реально находится в anchor;
- не подменяет неподтверждённую цитату;
- нормализует даты алгоритмически;
- не берёт срок новой задачи из соседней реплики;
- чистит ложные `Выполненные` и `Новые`;
- извлекает ответственного из evidence или speaker fallback;
- дедуплицирует уже валидированные строки;
- считает стабильность за 5 прогонов.

## Контроль ресурсов

LLM не получает весь транскрипт целиком. Предобработка сокращает вход:

| Файл | Реплик всего | Anchor-блоков | Уникальных реплик в anchors | Сокращение входа |
| --- | ---: | ---: | ---: | ---: |
| `transcript.txt` | 459 | 33 | 181 | 60.6% |
| `transcript2.txt` | 614 | 8 | 105 | 82.9% |
| `transcript3.txt` | 355 | 25 | 74 | 79.2% |

Это важно не только для стоимости API, но и для качества: чем меньше лишнего
контекста видит модель, тем меньше шанс, что она перенесёт срок или
ответственного из соседней темы.

## SQLite

При `--save-sqlite` сохраняются:

- `extraction_runs` - запуск, дата, пороги, итог стабильности;
- `meeting_tasks` - финальный результат и support-метаданные;
- `meeting_run_tasks` - строки каждого из 5 прогонов;
- `meeting_run_metrics` - counts по каждому run;
- `stability_report_lines` - текстовый отчёт стабильности.

Финальный `DataFrame` остаётся чистым, а audit-данные живут отдельно.

## Golden evaluation

Golden-разметка лежит в `tests/fixtures/golden/`.

Она нужна, чтобы измерять не только стабильность, но и качество:

- `precision` - сколько найденных строк действительно похоже на ожидаемые;
- `recall` - сколько ожидаемых задач найдено;
- `full_recall` - сколько задач найдено с правильным ответственным и сроком;
- список пропусков;
- список false positives;
- ошибки по полям.

## Структура проекта

```text
main.py                    CLI и orchestration
requirements.txt           основные зависимости
requirements-semantic.txt  optional SentenceTransformers/E5
src/config.py              настройки окружения
src/conversation_lexicon.py общий словарь разговорных маркеров
src/date_patterns.py       общие regex-паттерны дат
src/date_normalizer.py     нормализация сроков
src/preprocess.py          парсинг и anchors
src/prompts.py             prompt-контракты
src/schemas.py             dataclasses, DataFrame columns, JSON Schema
src/llm_client.py          OpenAI extraction и verifier
src/postprocess.py         validation, responsible, dedupe
src/stability.py           5-run метрики, consensus, verifier support
src/sqlite_store.py        запись результата в SQLite
src/golden_eval.py         сравнение с golden-разметкой
src/lexicon_audit.py       аудит словарей без LLM
src/semantic_similarity.py optional semantic similarity backend
tests/                     unit-тесты
tests/fixtures/golden/     ручная expected-разметка
trascripts/                входные транскрипты
```

## Проверки

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q .
git diff --check
```

Последняя локальная проверка:

```text
90 passed
```

## Что дальше

Ближайший технический фокус описан в [report.md](report.md), но коротко:

1. Ввести deterministic candidate registry до LLM-consensus.
2. Канонизировать формулировки задач до сравнения между прогонами.
3. Улучшить grouping anchors по связности, а не только пачками.
4. Использовать verifier как field-level audit, а не как способ спасать
   нестабильную генерацию.
5. После этого отдельно оценить, нужен ли E5/SentenceTransformers для
   similarity.
