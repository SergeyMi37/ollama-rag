import os, sys
import time
from datetime import timedelta
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
        self.query_stats = []  # Для хранения статистики запросов
        
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
        # Settings.сhunk_length_to_split_on = lambda doc: ()
        # Settings.length_function=len
        Settings.separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""] # Приоритет разделителей
        print('Settings configured')
    
    def _load_documents_and_build_index(self):
        """Загрузка документов и построение векторного индекса"""
        start_time = time.time()
        
        # Загрузка документов
        self.documents = SimpleDirectoryReader(self.docs_directory).load_data()
        load_time = time.time() - start_time
        
        print(f'Loaded {len(self.documents)} documents from {self.docs_directory} in {load_time:.2f} seconds')
        
        # Построение векторного индекса
        index_start_time = time.time()
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            llm=self.llm,
            embed_model=self.embed_model
        )
        index_time = time.time() - index_start_time
        
        print(f'Vector index built in {index_time:.2f} seconds')
    
    def query(
        self,
        prompt_text: str,
        temperature: float = 0,
        max_tokens: int = 2000,
        similarity_top_k: int = 3,
        timeout: float = 600.0,
        response_mode: ResponseMode = ResponseMode.COMPACT,
        verbose: bool = False
    ) -> dict:
        """
        Выполняет запрос к RAG системе с измерением скорости.
        
        Args:
            prompt_text (str): Текст промпта для запроса
            temperature (float): Температура для генерации
            max_tokens (int): Максимальное количество токенов в ответе
            similarity_top_k (int): Количество похожих фрагментов для поиска
            timeout (float): Таймаут выполнения запроса в секундах
            response_mode (ResponseMode): Режим формирования ответа
            verbose (bool): Выводить ли подробную информацию о скорости
        
        Returns:
            dict: Словарь с ответом и метриками производительности
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call _load_documents_and_build_index() first.")
        
        start_time = time.time()
        
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
        
        engine_creation_time = time.time() - start_time
        
        # Выполнение запроса
        query_start_time = time.time()
        response = query_engine.query(prompt_text)
        query_execution_time = time.time() - query_start_time
        
        total_time = time.time() - start_time
        
        # Сбор статистики
        query_info = {
            "prompt": prompt_text,
            "response": response.response,
            "metrics": {
                "total_time": total_time,
                "engine_creation_time": engine_creation_time,
                "query_execution_time": query_execution_time,
                "similarity_top_k": similarity_top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": time.time()
            }
        }
        
        self.query_stats.append(query_info)
        
        if verbose:
            self._print_query_stats(query_info)
        
        return query_info
    
    def _print_query_stats(self, query_info: dict):
        """Выводит статистику выполнения запроса"""
        metrics = query_info["metrics"]
        print(f"\n📊 Статистика выполнения запроса:")
        print(f"   🔍 Поиск похожих фрагментов: {metrics['similarity_top_k']}")
        print(f"   ⚙️  Создание движка: {metrics['engine_creation_time']:.2f} сек")
        print(f"   🤖 Выполнение запроса: {metrics['query_execution_time']:.2f} сек")
        print(f"   ⏱️  Общее время: {metrics['total_time']:.2f} сек")
        print(f"   📝 Длина ответа: {len(query_info['response'])} символов")
        print("-" * 60)
    
    def batch_query(self, prompts: list, verbose: bool = True, **kwargs) -> list:
        """
        Выполняет несколько запросов последовательно с измерением скорости.
        
        Args:
            prompts (list): Список промптов
            verbose (bool): Выводить ли подробную информацию
            **kwargs: Дополнительные параметры для query метода
        
        Returns:
            list: Список результатов с метриками
        """
        batch_start_time = time.time()
        results = []
        
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f'\n🔄 Обработка запроса {i+1}/{len(prompts)}')
            
            result = self.query(prompt, verbose=verbose, **kwargs)
            results.append(result)
        
        batch_time = time.time() - batch_start_time
        if verbose:
            print(f"\n🎯 Пакетная обработка завершена:")
            print(f"   📊 Обработано запросов: {len(prompts)}")
            print(f"   ⏱️  Общее время пакета: {batch_time:.2f} сек")
            print(f"   📈 Среднее время на запрос: {batch_time/len(prompts):.2f} сек")
        
        return results

    def get_system_info(self) -> dict:
        """Возвращает информацию о системе"""
        return {
            "ollama_url": self.ollama_url,
            "docs_directory": self.docs_directory,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "documents_loaded": len(self.documents) if hasattr(self, 'documents') else 0,
            "index_initialized": self.index is not None,
            "total_queries": len(self.query_stats)
        }
    
    def get_performance_stats(self) -> dict:
        """Возвращает общую статистику производительности"""
        if not self.query_stats:
            return {"message": "No queries executed yet"}
        
        total_queries = len(self.query_stats)
        total_time = sum(stat["metrics"]["total_time"] for stat in self.query_stats)
        avg_time = total_time / total_queries
        
        return {
            "total_queries": total_queries,
            "total_execution_time": total_time,
            "average_query_time": avg_time,
            "fastest_query": min(stat["metrics"]["total_time"] for stat in self.query_stats),
            "slowest_query": max(stat["metrics"]["total_time"] for stat in self.query_stats),
            "average_response_length": sum(len(stat["response"]) for stat in self.query_stats) / total_queries
        }
    
    def print_detailed_stats(self):
        """Выводит подробную статистику всех запросов"""
        if not self.query_stats:
            print("📊 Статистика: запросов еще не выполнено")
            return
        
        print("\n" + "="*80)
        print("📈 ДЕТАЛЬНАЯ СТАТИСТИКА ВЫПОЛНЕННЫХ ЗАПРОСОВ")
        print("="*80)
        
        for i, stat in enumerate(self.query_stats, 1):
            metrics = stat["metrics"]
            print(f"\n#{i} | Время: {metrics['total_time']:.2f} сек | Токены: {metrics['max_tokens']} | Temp: {metrics['temperature']}")
            print(f"   Промпт: {stat['prompt'][:100]}...")
            print(f"   Ответ: {stat['response'][:150]}...")
        
        perf_stats = self.get_performance_stats()
        print(f"\n{'='*80}")
        print("📊 СВОДНАЯ СТАТИСТИКА:")
        print(f"   Всего запросов: {perf_stats['total_queries']}")
        print(f"   Общее время: {perf_stats['total_execution_time']:.2f} сек")
        print(f"   Среднее время: {perf_stats['average_query_time']:.2f} сек")
        print(f"   Самый быстрый: {perf_stats['fastest_query']:.2f} сек")
        print(f"   Самый медленный: {perf_stats['slowest_query']:.2f} сек")
        print(f"   Средняя длина ответа: {perf_stats['average_response_length']:.0f} символов")

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
        prompt_text = sys.argv[3]
    else:
        # По умолчанию
        embedding_model="nomic-embed-text"
        llm_model="llama2"
        prompt_text = "Используя исходные документы, выбери номер темы, который больше всего подходит по частотному анализу слов для текста: 'Отсутствует водоснабжение'. В Ответ предоставь только номер темы."

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
    result1 = rag_system.query(
        prompt_text=prompt_text,
        temperature=0,
        max_tokens=1000
    )
    # Вывод общей статистики
    # rag_system.print_detailed_stats()

    print("Ответ 1:", result1["response"])
    print('Выполнялось: ',result1["metrics"]["query_execution_time"])
    
    # Второй запрос
    # result2 = rag_system.query(
    #     prompt_text="Кратко summarise основные темы из документов.",
    #     temperature=0.1,
    #     max_tokens=500
    # )
    # print("Ответ 2:", result2["response"])
    
    # # Третий запрос с другими параметрами
    # result3 = rag_system.query(
    #     prompt_text="Найди информацию о конкретных технологиях или методах.",
    #     similarity_top_k=5,
    #     temperature=0.2
    # )
    # print("Ответ 3:", result3["response"])
    
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
    
