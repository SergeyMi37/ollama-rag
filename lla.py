# Пример 2. RAG с LlamaIndex:
# !pip install llama-index > /dev/null

# RAG с LlamaIndex и Ollama (с локальными embedding-моделями):
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import OllamaEmbedding
from llama_index.llms import Ollama

# Настройка Ollama
OLLAMA_URL = "http://127.0.0.1:11434"

# Инициализируем embedding-модель из Ollama
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # или другая embedding-модель
    base_url=OLLAMA_URL
)

# Инициализируем LLM модель через Ollama
llm = Ollama(
    model="llama2",  # или другая модель
    base_url=OLLAMA_URL,
    request_timeout=60.0
)

# Создаем сервисный контекст
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

# Загружаем документы
documents = SimpleDirectoryReader('docs/').load_data()

# Строим векторный индекс
index = VectorStoreIndex.from_documents(
    documents, 
    service_context=service_context
)

# Выполняем запрос
query_engine = index.as_query_engine(response_mode="compact")
response = query_engine.query("Каково влияние климатических изменений на сельское хозяйство?")
print(response.response)