import os, sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import ResponseMode

class RAGSystem:
    def __init__(
        self,
        docs_directory: str = "docs/",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama2",
        ollama_url: str = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Инициализация RAG системы с возможностью многократных запросов.
        
        Args:
            docs_directory (str): Путь к директории с документами
            embedding_model (str): Название модели для эмбеддингов
            llm_model (str): Название LLM-модели
            ollama_url (str): URL Ollama сервера
            chunk_size (int): Размер чанков для обработки документов
            chunk_overlap (int): Перекрытие между чанками
        """
        # Настройка Ollama URL
        if ollama_url is None:
            ollama_url = os.environ.get('OLLAMA_URL', "http://127.0.0.1:11434")
        
        self.ollama_url = ollama_url
        self.docs_directory = docs_directory
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        
        print(f'Ollama URL: {ollama_url}')
        
        # Инициализация моделей
        self._initialize_models(embedding_model, llm_model, chunk_size, chunk_overlap)
        
        # Загрузка документов и построение индекса
        self._load_documents_and_build_index()
    
    def _initialize_models(self, embedding_model: str, llm_model: str, chunk_size: int, chunk_overlap: int):
        """Инициализация embedding модели и LLM"""
        # Инициализация embedding-модели
        self.embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=self.ollama_url,
            timeout=120.0
        )
        print(f'Embedding model: {embedding_model} initialized')
        
        # Инициализация LLM
        self.llm = Ollama(
            model=llm_model,
            base_url=self.ollama_url,
            request_timeout=120.0
        )
        print(f'LLM model: {llm_model} initialized')
        
        # Настройка глобальных параметров
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        print('Settings configured')
    
    def _load_documents_and_build_index(self):
        """Загрузка документов и построение векторного индекса"""
        # Загрузка документов
        self.documents = SimpleDirectoryReader(self.docs_directory).load_data()
        print(f'Loaded {len(self.documents)} documents from {self.docs_directory}')
        
        # Построение векторного индекса
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            llm=self.llm,
            embed_model=self.embed_model
        )
        print('Vector index built')
    
    def query(
        self,
        prompt_text: str,
        temperature: float = 0,
        max_tokens: int = 2000,
        similarity_top_k: int = 3,
        timeout: float = 600.0,
        response_mode: ResponseMode = ResponseMode.COMPACT
    ) -> str:
        """
        Выполняет запрос к RAG системе.
        
        Args:
            prompt_text (str): Текст промпта для запроса
            temperature (float): Температура для генерации
            max_tokens (int): Максимальное количество токенов в ответе
            similarity_top_k (int): Количество похожих фрагментов для поиска
            timeout (float): Таймаут выполнения запроса в секундах
            response_mode (ResponseMode): Режим формирования ответа
        
        Returns:
            str: Ответ модели
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call _load_documents_and_build_index() first.")
        
        # Создание query engine с конкретными параметрами для этого запроса
        query_engine = self.index.as_query_engine(
            response_mode=response_mode,
            timeout=timeout,
            similarity_top_k=similarity_top_k,
            streaming=False,
            llm_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        print(f'Executing query: {prompt_text[:100]}...')
        response = query_engine.query(prompt_text)
        print('Query executed')
        
        return response.response
    
    def batch_query(self, prompts: list, **kwargs) -> list:
        """
        Выполняет несколько запросов последовательно.
        
        Args:
            prompts (list): Список промптов
            **kwargs: Дополнительные параметры для query метода
        
        Returns:
            list: Список ответов
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f'Processing prompt {i+1}/{len(prompts)}')
            result = self.query(prompt, **kwargs)
            results.append(result)
        return results

    def get_system_info(self) -> dict:
        """Возвращает информацию о системе"""
        return {
            "ollama_url": self.ollama_url,
            "docs_directory": self.docs_directory,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "documents_loaded": len(self.documents) if hasattr(self, 'documents') else 0,
            "index_initialized": self.index is not None
        }

# Функция для быстрого создания системы (обратная совместимость)
def create_rag_system(
    docs_directory: str = "docs/",
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "llama2",
    ollama_url: str = None
) -> RAGSystem:
    """
    Создает и возвращает готовую RAG систему.
    """
    return RAGSystem(
        docs_directory=docs_directory,
        embedding_model=embedding_model,
        llm_model=llm_model,
        ollama_url=ollama_url
    )

# Примеры использования:
if __name__ == "__main__":
    # Если вызвано из командной строки с аргументами
    if len(sys.argv) > 2:
        embedding_model = sys.argv[1]
        llm_model = sys.argv[2]
    else:
        # По умолчанию
        embedding_model="nomic-embed-text"
        llm_model="llama2"

    # Инициализация системы один раз
    print("=== Инициализация RAG системы ===")
    rag_system = RAGSystem(
        embedding_model=embedding_model, 
        llm_model=llm_model
    )
    
    # Проверка информации о системе
    info = rag_system.get_system_info()
    print("Системная информация:", info)
    
    # print("\n=== Выполнение нескольких запросов ===")
    # Первый запрос
    response1 = rag_system.query(
        prompt_text="Используя исходные документы, выбери номер темы, который больше всего подходит по частотному анализу слов для текста: 'Отсутствует водоснабжение'. Ответ предоставь на русском языке.",
        temperature=0,
        max_tokens=1000
    )
    print("Ответ 1:", response1)
    print("-" * 50)
    
    # # Второй запрос
    # response2 = rag_system.query(
    #     prompt_text="Кратко summarise основные темы из документов.",
    #     temperature=0.1,
    #     max_tokens=500
    # )
    # print("Ответ 2:", response2)
    # print("-" * 50)
    
    # # Третий запрос с другими параметрами
    # response3 = rag_system.query(
    #     prompt_text="Найди информацию о конкретных технологиях или методах.",
    #     similarity_top_k=5,
    #     temperature=0.2
    # )
    # print("Ответ 3:", response3)
    # print("-" * 50)
    
    # # Пакетная обработка запросов
    # print("\n=== Пакетная обработка ===")
    # prompts = [
    #     "Какие основные проблемы обсуждаются в документах?",
    #     "Какие решения предлагаются?",
    #     "Кто является целевой аудиторией документов?"
    # ]
    
    # batch_results = rag_system.batch_query(
    #     prompts,
    #     temperature=0.1,
    #     max_tokens=800
    # )
    
    # for i, result in enumerate(batch_results):
    #     print(f"Пакетный ответ {i+1}: {result[:200]}...")
    #     print("-" * 50)