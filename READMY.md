# activate the environment
optional, if it need in Windows:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
required:
.venv/Scripts/Activate.ps1


.\.venv\Scripts\python.exe main.py

.\.venv\Scripts\python.exe main.py -q "Приведи пример интерпретации логитов как соотношения шансов для логистической регрессии для предсказания, например, цены на земельный участок"



# normalized text
python src/parse_introchem_xhtml_v3.py `
  --input "data/raw/introductory_chemistry/Introductory-Chemistry-1st-Canadian-Edition-1695676481.html" `
  --chapter 18 `
  --output "data/normalized/introductory_chemistry/chapter_18_v3"

# chunk for rag creation
python src/build_introchem_rag_chunks_v3.py `
  --normalized-root "data/normalized/introductory_chemistry" `
  --output "data/rag/introductory_chemistry_v3"

# index creation
python src/introchem_vector_search.py build --chunks "data/rag/introductory_chemistry/rag_chunks.jsonl" --db "data/indexes/introductory_chemistry_chroma" --collection "introchem_theory_v1" --model "embeddinggemma" --rebuild


# query
#### one question
python src/introchem_vector_search.py search `
  --query "What is a chemical equation and why must it be balanced?" `
  -k 5


#### interactive mode

python src/introchem_vector_search.py interactive

example:
Запрос: What is oxidation?
Запрос: How is an ionic bond formed?
Запрос: Що таке відновник?
Запрос: /exit

#### mass checking
python src/introchem_vector_search.py batch `
  --tests "config/introchem_retrieval_queries.jsonl" `
  --output "data/rag/introductory_chemistry/retrieval_review.md" `
  -k 5
  
#### only one chapter search
python src/introchem_vector_search.py search `
  --query "How do oxidation numbers change?" `
  --chapter 14



#### only theory
python src/introchem_vector_search.py search `
  --query "What is a reducing agent?" `
  --retrieval-group theory

#### and theory and self-checking questions
python src/introchem_vector_search.py search `
  --query "Answer to the self-test about balancing ammonia formation" `
  --include-nondefault

#### data about index
python src/introchem_vector_search.py info
