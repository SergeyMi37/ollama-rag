# ollama-RAG   🍁
# Спасибо автору https://habr.com/ru/articles/931396/

# В этом коде мы: загружаем все .txt файлы из папки, режем их на фрагменты по ~1000 символов, получаем для них эмбеддинги через OllamaEmbedding, сохраняем в локальный индекс VectorStoreIndex, затем при запросе извлекаем топ-3 похожих фрагмента и передаем их вместе с вопросом в модель Ollama. Результат – сгенерированный ответ. В реальном сценарии, вместо печати, можно обернуть это в веб-сервис или чат-интерфейс.


``` bash
git clone https://github.com/SergeyMi37/ollama-rag
cd ollama-rag
```

Create virtual environment (optional)
``` bash
python3 -m venv env
source env/bin/activate
```

Create virtual environment for Windows
``` bash
python -m venv env
source env/Scripts/activate
```

Install all requirements:
``` bash
pip install -r requirements.txt
```

Выполнить запрос к нейронке
``` bash
python llama-core.py # embedding_model="nomic-embed-text"   llm_model="llama2"
python llama-rag.py "mxbai-embed-large" "llama3.1" # Согласно источникам, наиболее подходящий номер темы, который соответствует данной фразе, — это 2.2.
python llama-rag.py "mxbai-embed-large" "infidelis/GigaChat-20B-A3B-instruct-v1.5:q5_0" # Текст "Отсутствует водоснабжение" соответствует теме с номером **2.2**.
```
Выполнить запрос к нейронке с параметрами
``` bash
$ python llama-rag-stat.py "mxbai-embed-large" "llama3.1" "Используя исходные документы, 
выбери номер темы, который больше всего подходит по частотному анализу слов для текста: ' 
Отсутствует водоснабжение'. В Ответ предоставь только номер темы."
```
Системная информация: {'ollama_url': 'http://127.0.0.1:11434', 'docs_directory': 'docs/', 'embedding_model': 'mxbai-embed-large', 'llm_model': 'llama3.1', 'documents_loaded': 1, 'index_initialized': True, 'total_queries': 0}
Ответ 1: 2.2
Выполнялось:  12.457962989807129

python llama-rag-stat.py "all-minilm" "mistral" "Используя исходные документы, выбери номер темы, котор
ый больше всего подходит по частотному анализу слов для текста: 'Отсутствует водоснабжение'. Ответ предос 
тавь только номер темы."
=== Инициализация RAG системы ===
Ollama URL: http://127.0.0.1:11434
Embedding model: all-minilm initialized
LLM model: mistral initialized
Settings configured
Loaded 1 documents from docs/ in 0.03 seconds
Vector index built in 0.23 seconds
Системная информация: {'ollama_url': 'http://127.0.0.1:11434', 'docs_directory': 'docs/', 'embedding_model': 'all-minilm', 'llm_model': 'mistral', 'documents_loaded': 1, 'index_initialized': True, 'total_queries': 0}
Ответ 1:  2.2 (отсутствие водоснабжения)
Выполнялось:  1.5262219905853271