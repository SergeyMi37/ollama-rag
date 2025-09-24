# Альтернатива с LangChain и Ollama embeddings
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain_ollama import OllamaEmbeddings

# Настройка Ollama
OLLAMA_URL = "http://127.0.0.1:11434"

# Инициализируем embedding-модель через LangChain
embed_model = OllamaEmbeddings(
    model="nomic-embed-text",  # или другая embedding-модель
    base_url=OLLAMA_URL
)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Загружаем документы и строим индекс
documents = SimpleDirectoryReader('docs/').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Выполняем запрос
query_engine = index.as_query_engine()
response = query_engine.query("Каково влияние климатических изменений на сельское хозяйство?")
print(response.response)