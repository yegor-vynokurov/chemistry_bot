# Prompt Garden v5

Локальный мини-PromptOps без серверов.

## Новое в v5

- Автоматическая генерация всех `system × user` combos.
- Combo получает `test_status="untested"`.
- Эксперимент стал отдельным узлом графа.
- Эксперимент создаётся заранее и требует:
  - `name`
  - `goal`
  - `hypothesis`
- Связи `experiment → combo` пишутся в `registry/edges.jsonl`.
- Полный объект эксперимента хранится в `experiments/exp_*.json`.
- У prompt nodes появились stats:
  - chars
  - words
  - sentences
  - average word length
  - average sentence length
  - placeholders
- Результаты эксперимента могут хранить:
  - automatic score
  - subjective score
  - human result text
  - case results
  - prompt stats snapshot

## Запуск

```bash
jupyter notebook prompt_garden_experiments_control.ipynb
```

Для запуска LLM-тестов нужен Ollama и зависимости химического бота:

```bash
pip install langchain-ollama langchain-core pydantic
```

## Главный сценарий

```python
created = garden.generate_combos()
experiment = garden.create_experiment(
    name="...",
    goal="...",
    hypothesis="...",
)
garden.attach_combos_to_experiment(
    experiment["id"],
    [combo["id"] for combo in garden.list_untested_combos()],
)
```

Потом notebook запускает combo-тесты в цикле и записывает результаты в эксперимент.
