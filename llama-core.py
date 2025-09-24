# Пример 2. RAG с LlamaIndex:
# !pip install llama-index > /dev/null

# pip uninstall llama-index
# pip install llama-index-core
# pip install llama-index-llms-ollama
# pip install llama-index-embeddings-ollama

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.response_synthesizers import ResponseMode

# Настройка Ollama
OLLAMA_URL = "http://127.0.0.1:11434" if not os.environ.get('OLLAMA_URL', default=False) else os.environ.get('OLLAMA_URL')
print('----0 ',OLLAMA_URL)
# Инициализируем embedding-модель из Ollama
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
    timeout=120.0  # Таймаут для embedding-модели
)
print('----1')
llm = Ollama(
    model="llama2",
    base_url=OLLAMA_URL,
    request_timeout=120.0
)
print('----2')
Settings.llm = llm  # Например, ваша модель Ollama
Settings.embed_model = embed_model  # Например, ваша модель эмбеддингов Ollama
Settings.chunk_size = 1000
Settings.chunk_overlap = 100
print('----3')

# Загружаем документы
documents = SimpleDirectoryReader('docs/').load_data()
print('----4',documents)

# Строим векторный индекс
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,  # Передача LLM
    embed_model=embed_model  # Передача модели эмбеддингов
)
print('----5')
# Выполняем запрос
query_engine = index.as_query_engine(
    response_mode=ResponseMode.COMPACT,
    timeout=600.0,  # 10 минут
    similarity_top_k=3,  # Количество похожих фрагментов
    streaming=False,
    llm_kwargs={
        "temperature": 0,
        "max_tokens": 2000
    }
    )
print('----6')
# prompt  = "Используя исходные документы RAG, напиши промпт для генерации кода на Python для решения следующей задачи:" \
# " Напи программу, которая будет выводить на экран текст 'Hello, World!' использую библитеку kivy."
prompt  = "Используя исходные документы, выбери номер темы, который больше все подходит по частотному анализу слов для текста: 'Отсутствует водоснабжение'. Ответ предоставь на русском." 

response = query_engine.query(prompt)

print('----7')
print(response.response)
