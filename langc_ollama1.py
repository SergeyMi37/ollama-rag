!pip install langchain faiss-cpu huggingface_hub ollama

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManagerForLLMCallback
from langchain.llms.base import BaseLLM
from typing import Any, Dict, Optional
from pydantic import Field
from requests import post as rpost


# Локальная реализация лангчейна для работы с ollama
class Ollama(BaseLLM):
    """Класс для интеграции с ollama."""
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "llama2", # Модель на вашем сервере ollama
            "prompt": prompt,
            "stream": False,
        }
        response = rpost("http://localhost:11434/api/generate", headers=headers, json=payload)
        answer = response.json().get('response', '')
        return answer.strip()

    async def _acall(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async mode not implemented yet.")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

# Загрузка документов из каталога
loader = DirectoryLoader("docs/", glob="*.txt")
documents = loader.load()

# Разделение документов на чанки (~1000 символов с перекрытием 100)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_chunks = text_splitter.split_documents(documents)

# Создание эмбеддингов и сохранение в векторном хранилище FAISS
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(docs_chunks, embedding=embeddings)

# Настройка ретривера для поиска похожих документов
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Генерация цепочки Вопрос-Ответ (Question Answering Chain)
llm = Ollama()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Запускаем тестовый запрос
query = "Что говорится в документах о влиянии климатических изменений на сельское хозяйство?"
result = qa_chain.run(query)
print(result)
