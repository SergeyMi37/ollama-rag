import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import ResponseMode

def rag_query(
    prompt_text: str,
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "llama2",
    docs_directory: str = "docs/",
    ollama_url: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    temperature: float = 0,
    max_tokens: int = 2000,
    similarity_top_k: int = 3,
    timeout: float = 600.0
):
    """
    Выполняет RAG-запрос с использованием указанных моделей Ollama.
    
    Args:
        prompt_text (str): Текст промпта для запроса
        embedding_model (str): Название модели для эмбеддингов
        llm_model (str): Название LLM-модели
        docs_directory (str): Путь к директории с документами
        ollama_url (str): URL Ollama сервера (None для автоматического определения)
        chunk_size (int): Размер чанков для обработки документов
        chunk_overlap (int): Перекрытие между чанками
        temperature (float): Температура для генерации
        max_tokens (int): Максимальное количество токенов в ответе
        similarity_top_k (int): Количество похожих фрагментов для поиска
        timeout (float): Таймаут выполнения запроса в секундах
    
    Returns:
        str: Ответ модели
    """
    
    # Настройка Ollama URL
    if ollama_url is None:
        ollama_url = os.environ.get('OLLAMA_URL', "http://127.0.0.1:11434")
    
    print(f'Ollama URL: {ollama_url}')
    
    # Инициализация embedding-модели
    embed_model = OllamaEmbedding(
        model_name=embedding_model,
        base_url=ollama_url,
        timeout=120.0
    )
    print(f'Embedding model: {embedding_model} initialized')
    
    # Инициализация LLM
    llm = Ollama(
        model=llm_model,
        base_url=ollama_url,
        request_timeout=120.0
    )
    print(f'LLM model: {llm_model} initialized')
    
    # Настройка глобальных параметров
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    print('Settings configured')
    
    # Загрузка документов
    documents = SimpleDirectoryReader(docs_directory).load_data()
    print(f'Loaded {len(documents)} documents from {docs_directory}')
    
    # Построение векторного индекса
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model
    )
    print('Vector index built')
    
    # Создание query engine
    query_engine = index.as_query_engine(
        response_mode=ResponseMode.COMPACT,
        timeout=timeout,
        similarity_top_k=similarity_top_k,
        streaming=False,
        llm_kwargs={
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    )
    print('Query engine created')
    
    # Выполнение запроса
    response = query_engine.query(prompt_text)
    print('Query executed')
    
    return response.response

# Примеры использования:
if __name__ == "__main__":
    # Пример 1: Базовый вызов с параметрами по умолчанию
    # result1 = rag_query(
    #     prompt_text="Используя исходные документы, выбери номер темы, который больше всего подходит по частотному анализу слов для текста: 'Отсутствует водоснабжение'. Ответ предоставь на русском."
    # )
    # print("Результат 1:", result1)
    
    # Пример 2: Вызов с кастомными моделями и параметрами
    result2 = rag_query(
        prompt_text="Проанализируй документы и составь краткое содержание.",
        embedding_model="nomic-embed-text",
        #llm_model="mistral",
        temperature=0.1,
        max_tokens=1000,
        similarity_top_k=5
    )
    print("Результат 2:", result2)
    
    # Пример 3: Вызов с указанием кастомного URL
    # result3 = rag_query(
    #     prompt_text="Найди информацию о конкретной технологии.",
    #     ollama_url="http://localhost:11434",
    #     docs_directory="./my_documents/"
    # )
    # print("Результат 3:", result3)