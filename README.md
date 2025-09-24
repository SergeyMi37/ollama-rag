# ollama-RAG   🍁
# Спасибо автору https://habr.com/ru/articles/931396/
# Пример 1. RAG с LangChain (векторное хранилище FAISS + OpenAI):
# В этом коде мы: загружаем все .txt файлы из папки, режем их на фрагменты по ~1000 символов, получаем для них эмбеддинги через OpenAI, сохраняем в локальный индекс FAISS, затем при запросе извлекаем топ-3 похожих фрагмента и передаем их вместе с вопросом в модель GPT-3.5. Цепочка RetrievalQA (тип stuff) просто «подкладывает» все найденные тексты в промпт. Результат (result) – сгенерированный ответ. В реальном сценарии, вместо печати, можно обернуть это в веб-сервис или чат-интерфейс.
# Пример 2. RAG с LlamaIndex:

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

Укажите исходную и целевую директорию
``` bash
python llama-core.py # embedding_model="nomic-embed-text"   llm_model="llama2"
python llama-rag.py "mxbai-embed-large" "llama3.1" # Согласно источникам, наиболее подходящий номер темы, который соответствует данной фразе, — это 2.2.
python llama-rag.py "mxbai-embed-large" "infidelis/GigaChat-20B-A3B-instruct-v1.5:q5_0" # Текст "Отсутствует водоснабжение" соответствует теме с номером **2.2**.
```

