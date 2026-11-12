# README2 · запуск локального Compose-прототипа

Инструкция для локального Chemistry Tutor и AnythingLLM. Она дополняет [README](README.md) и техническое описание [Compose-прототипа](docs/compose-prototype.md).

> **Состояние на 2026-11-12.** Main Compose-контур на этой машине отвечает и оба контейнера здоровы, но его текущий API моделей возвращает только совместимый ID `chemistry-tutor`: основной runtime ещё не пересобирался на новом commit. В локальной ветке `change/git-model-bootstrap` создан commit `b201d23c83403f21380777f0aef9526cd82d2994`; точный commit проверен из отдельного чистого локального Git clone по C1–C3. Ветка не отправлялась в remote; опубликованный clone не проверялся.

## Быстрый старт

Нужны Docker Desktop с Linux-контейнерами и доступ к Docker Hub/GHCR и Ollama model registry. Host Ollama необязательна. Из PowerShell в корне рабочей копии укажи точный тег модели:

```powershell
.\scripts\start-compose.ps1 -Model gemma4:e2b-it-q4_K_M
```

Список моделей тьютора в этой ревизии:

- `gemma4:e2b-it-q4_K_M` (добавлена этим Change; использовалась в qualification run);
- `gemma4:12b`
- `phi4-mini:latest`
- `phi4-mini-4k:latest`
- `qwen3.5:9b-q4_K_M`

Список и вспомогательные модели задаются в `src/chemistry_bot/integrations/anythingllm_probe/tutor_models.json`. AnythingLLM может выбрать только модель, установленную в активном backend; новую модель загружают повторным запуском команды с нужным `-Model`, а не во время чата.

## Выбор Ollama и модели

Launcher проверяет, отвечает ли host Ollama и установлены ли там выбранная модель вместе со всеми требуемыми вспомогательными моделями. Если всё найдено, используется host Ollama. Иначе запускается Docker Ollama, и выбранная модель вместе с недостающими вспомогательными моделями загружается в постоянный каталог:

```text
data/compose-poc/ollama
```

Docker fallback не устанавливает и не загружает модели в host Ollama. Для проверки изолированного Docker backend его можно выбрать явно:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest -ForceDockerOllama
```

Размер загрузки зависит от точного тега и уже имеющихся слоёв. Перед первым Docker fallback проверь свободное место на диске.

Launcher и tutor в commit `b201d23` поддерживают выбор среди доступных разрешённых моделей. В Qualification C1 выбранная модель отсутствовала в отвечающей host Ollama, поэтому launcher запустил Docker Ollama и загрузил Gemma в её постоянное хранилище; C2 подтвердил выбор Gemma в AnythingLLM и ответ с источником. Текущие main-контейнеры этой машины ещё не пересобраны на commit `b201d23`, поэтому их `/v1/models` всё ещё возвращает только совместимый ID `chemistry-tutor`.

После запуска открой <http://localhost:3001/>. Порт AnythingLLM привязан только к loopback. При первом старте хранилище пустое: создай workspace и загрузи свой документ.

## TLS и локальная CA

Если сеть проверяет TLS своим локальным CA, передай путь к публичному сертификату:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest -CaFile 'D:\path\to\local-root.crt'
```

TLS-проверка остаётся включённой. Для сборки сертификат передаётся как временный BuildKit secret. При Docker fallback отдельная копия CA монтируется в Ollama только для чтения и устанавливается в runtime до запуска сервера. Она нужна, пока Docker Ollama потребуется пересоздавать; после остановки этого backend ненужную копию можно удалить вручную из `data/compose-poc/trust/docker-ollama-runtime-ca.crt`. CA не входит в image или Git. При обычной системной цепочке доверия `-CaFile` не нужен.

## Проверка ответа с источником

1. В AnythingLLM создай workspace.
2. Для чата выбери режим **Query**.
3. Загрузи и добавь как embedded document `tests/fixtures/compose-poc/oxygen_atomic_number.md` или свой учебный документ.
4. Спроси: «Каков атомный номер кислорода и сколько у него протонов?»
5. Проверь ответ и source metadata AnythingLLM: источник должен совпадать с загруженным документом.

Фикстура содержит краткий пересказ и ссылки на разделы OpenStax Chemistry 2e; это не прямая цитата из учебника. Проверка показывает связь ответа с загруженным документом, но сама по себе не подтверждает буквальную точность цитаты из OpenStax.

## Остановка, перезапуск и данные

Данные хранятся в bind-каталогах проекта:

- `data/compose-poc/anythingllm` — состояние AnythingLLM, документы, история и индекс;
- `data/compose-poc/tutor` — SQLite-состояние и история тьютора;
- `data/compose-poc/ollama` — модели Docker Ollama.

Остановить контейнеры, сохранив данные:

```powershell
docker compose --profile docker-ollama stop
```

Для пользовательского `-ProjectName NAME` указывай тот же проект:

```powershell
docker compose --project-name NAME --profile docker-ollama stop
```

Запускай контур и задавай backend/модель через launcher:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest
```

Не используй для обычного перезапуска голую команду `docker compose up --force-recreate`: выбор backend и модели задаётся launcher переменными процесса и не сохраняется после завершения команды. Если нужно удалить контейнеры, но оставить bind-каталоги, используй `docker compose --profile docker-ollama down` (с тем же `--project-name`, если он применялся). Не используй `down -v`, глобальный `docker system prune` или удаление `data/compose-poc/`, если нужны документы, история, индексы и модели.

## Что фактически квалифицировано

В Change `CHG-20261111-REPRODUCIBLE-STARTUP` записаны executor-claimed PASS для C1–C5: повторное использование host Ollama и пустой старт; Docker fallback с TLS; выбор модели, ответ с источником и история; сохранность источника, индекса, истории, модели и CA после пересоздания; список только моделей активного backend и отклонение недоступной модели до вызова графа. Отчёты ограничены Windows source snapshots; 8 сфокусированных unit tests также отмечены как пройденные в тех записях.

Change `CHG-20261112-WINDOWS-GIT-STARTUP` дополнительно квалифицировал на точном локальном commit `b201d23` чистый локальный clone, Docker Ollama fallback с `gemma4:e2b-it-q4_K_M`, ответ AnythingLLM с фактическими source metadata и сохранение модели, документа, retrieval index и истории после `--force-recreate`. C3 helper выдал ложный отрицательный флаг для сравнения model-store inventory; отдельная сверка сохранённых инвентарей подтвердила идентичные файлы и манифест до/после. Полные RUN и evidence находятся в `.planning/project_docs/tests/TEST-20261112-WINDOWS-GIT-STARTUP/`.

Это не независимый итоговый review и не проверка опубликованного clone: тестовый clone создан с локального репозитория. Docker images были уже закэшированы, поэтому их холодная загрузка не проверялась. Main-контейнеры не пересобирались и по-прежнему показывают только `chemistry-tutor`; qualification runs в этом README-проверочном проходе не перезапускались.

## Текущие границы

- Compose-прототип локальный, не production-развёртывание.
- Provider selection через AnythingLLM и мост тьютора пока ограничен проверенным локальным Ollama путём; OpenAI и Claude не квалифицированы.
- Prompt Garden имеет интерфейс и сценарии экспериментов для классического RAG-пути, но его реестр и статистика не подключены к активному Compose tutor graph.
- Сохранение истории диалога реализовано; долговременная структурированная оценка освоения, адаптивная постепенность и профиль ученика ещё не подтверждены как работающие способности.
- Чистый clone, другая ОС, холодный Docker cache, визуальная проверка выбора модели, backup/restore и multi-user режим не квалифицированы.

Срез фактического состояния и ближайший приоритет описаны в [.planning Reality snapshot](.planning/project_docs/02_real_state/overview.md) и [реконструкции проекта](.planning/project_docs/02_real_state/project_reconstruction.md).
