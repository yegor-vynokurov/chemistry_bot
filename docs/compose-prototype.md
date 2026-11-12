# Локальный Compose-прототип

Проект запускает интерфейс AnythingLLM и HTTP-мост Chemistry Tutor через Docker Compose. AnythingLLM доступен на `http://localhost:3001`; порт слушает только loopback. Модельный сервер может быть локальной Ollama на компьютере или отдельной Ollama в Compose.

## Запуск

Нужны Docker Desktop с Linux-контейнерами и доступ к Docker Hub/GHCR и Ollama model registry. Host Ollama необязательна.

В PowerShell из корня проекта:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest
```

Можно передать один из разрешённых начальных тегов:

- `gemma4:12b`
- `phi4-mini:latest`
- `phi4-mini-4k:latest`
- `qwen3.5:9b-q4_K_M`

Перечень задаётся в `src/chemistry_bot/integrations/anythingllm_probe/tutor_models.json`. Фиксированные модели вспомогательных ролей указаны там же: `phi4-mini:latest` для классификации/связи реплик/follow-up и `embeddinggemma:latest` для embeddings и RAG relevance.

Запускатель проверяет host Ollama и `/api/tags`. Если сервер отвечает и выбранная модель вместе со всеми фиксированными моделями уже установлена, Compose использует host endpoint через `host.docker.internal`. Иначе включается профиль `docker-ollama`: контейнер `ollama/ollama:0.31.2` получает только выбранный tutor tag и нужные support tags. Его постоянный bind-каталог — `data/compose-poc/ollama`; модели host Ollama не изменяются. Переключатель `-ForceDockerOllama` принудительно выбирает изолированный backend.

Например, принудительно использовать Docker Ollama:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest -ForceDockerOllama
```

AnythingLLM получает Generic OpenAI endpoint Chemistry Tutor. В квалифицированном source snapshot endpoint `/v1/models` показывал установленные разрешённые chat-модели и скрывал embedding/support модели; настройки AnythingLLM позволяли выбрать установленную модель. Если модель отсутствует, повтори startup command с её точным тегом. Во время обычного чата модели не загружаются. **Main-контейнер, проверенный 2026-11-12, возвращал только совместимый ID `chemistry-tutor`**; динамический каталог на нём сейчас не подтверждён.

### TLS и локальная CA

Сборка использует системный CA bundle и стандартную проверку сертификатов. Если сеть инспектирует TLS, можно передать публичный локальный CA-файл:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest -CaFile 'D:\path\to\local-root.crt'
```

Запускатель временно копирует указанный публичный CA под уникальным именем в игнорируемый `data/compose-poc/trust`, чтобы Windows Compose/Buildx мог прочитать файл внутри build context. В сборке файл доступен только как BuildKit secret; эта уникальная копия и временный Compose override удаляются после команды.

При Docker fallback отдельная копия `data/compose-poc/trust/docker-ollama-runtime-ca.crt` монтируется в Ollama read-only. Базовый `compose.yaml` устанавливает её в runtime до `ollama serve`. Обычная TLS-проверка остаётся включённой. Runtime-копия остаётся на диске, пока Docker Ollama можно пересоздать, и должна быть удалена пользователем после остановки этого backend, если она больше не нужна. В host-only режиме заново созданная запускателем runtime-копия удаляется в конце запуска. CA не включается в image или Git. Без CA и без системного доверия проверка сертификата должна завершиться ошибкой; отключать её нельзя. При обычной работе запускай или восстанавливай контур через `scripts/start-compose.ps1`: backend и выбранная модель передаются launcher как переменные текущего процесса. Голый `docker compose up --force-recreate` этих настроек не восстанавливает. На Windows-машине с Avast правила разовой подготовки локальной CA описаны ниже.

### Первое состояние AnythingLLM

Новая папка `data/compose-poc/anythingllm` означает пустой экземпляр. Создай workspace вручную, выбери для его чата режим **Query**, затем добавь документ. Для короткой проверки можно использовать `tests/fixtures/compose-poc/oxygen_atomic_number.md`; он не загружается автоматически.

## Хранилища и пересоздание

Все bind-каталоги находятся в проекте:

- `data/compose-poc/anythingllm` — настройки/база AnythingLLM, документы, LanceDB embeddings и история;
- `data/compose-poc/tutor` — SQLite состояние и история тьютора;
- `data/compose-poc/ollama` — persistent model store только Docker Ollama;
- `data/compose-poc/trust/docker-ollama-runtime-ca.crt` — исходная CA-копия для пересоздания Docker Ollama при TLS-inspection; до первого старта launcher создаёт пустой marker-файл, когда CA не задана.

Остановка с сохранением данных:

```powershell
docker compose --profile docker-ollama stop
```

Перезапуск:

```powershell
.\scripts\start-compose.ps1 -Model phi4-mini:latest
```

Для обычного перезапуска или смены модели повтори команду запуска через `scripts/start-compose.ps1`, указав нужный тег и при необходимости тот же `-CaFile`. Она повторно применяет выбранный backend и конфигурацию модели. Системное принудительное пересоздание с проверкой сохранности описано отдельными RUN в `.planning/project_docs/tests/TEST-20261111-REPRODUCIBLE-STARTUP/`; не заменяй его голым `docker compose up --force-recreate`, который не получает настройки launcher. Не используй `docker compose down -v`, `docker system prune` и не удаляй `data/compose-poc/`, если нужно сохранить состояние.

## Сборка на TLS-инспектируемом Avast-хосте

На текущем Windows-хосте Avast может выдавать сертификаты от локального корня, которого нет в Linux image. CA передаётся параметром `-CaFile` как BuildKit secret; при Docker fallback та же CA-копия монтируется в Ollama только для чтения и добавляется в runtime trust перед запуском сервера. TLS verification остаётся включённой и при пересоздании контейнера. При свежем clone на таком хосте экспортируй только публичный сертификат из Windows trust store в локальный файл, затем укажи его путь в параметре. Не добавляй CA или сгенерированный Compose override в Git.

Для старой ручной настройки также поддерживается игнорируемый `compose.override.yaml` с secret ID `avast_webshield_root`; Dockerfile сохраняет этот ID для совместимости. Запускатель без `-CaFile` оставляет автоматическую загрузку локального Compose override включённой.

## Границы прототипа и квалификации

Текущая квалификация описывается в `.planning/project_docs/changes/CHG-20261111-REPRODUCIBLE-STARTUP/PLAN.md` и `.planning/project_docs/tests/TEST-20261111-REPRODUCIBLE-STARTUP/`. Она проводится на изолированном source snapshot и не должна называться fresh Git clone или опубликованной ревизией. Production, другие ОС, холодный image cache, backup/restore, visual UI testing, cloud providers и multi-user режим остаются вне этой проверки.
