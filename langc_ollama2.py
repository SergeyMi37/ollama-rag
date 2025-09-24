# !pip install langchain faiss-cpu ollama > /dev/null
# pip install -U langchain-community

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMCallback
from typing import Any, Dict, Optional
from pydantic import Field

# Укажите адрес удалённого сервера Ollama
OLLAMA_URL = "http://127.0.0.1:11434"

# 1. Загружаем документы из директории
loader = DirectoryLoader("docs/", glob="*.txt")
documents = loader.load()

# 2. Делим документы на куски (~1000 символов с перекрытием 100)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_chunks = text_splitter.split_documents(documents)

# 3. Генерируем эмбеддинги с помощью удалённой Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
vector_store = FAISS.from_documents(docs_chunks, embedding=embeddings)

# 4. Создаем ретривер на основе векторного хранилища
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. Подготавливаем LLM на основе удалённого экземпляра Ollama
llm = ChatOllama(model="llama2", base_url=OLLAMA_URL, temperature=0)

# 6. Строим цепочку Question-Answering
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 7. Проверяем работоспособность с примером запроса
query = "Что говорится в документах о влиянии климатических изменений на сельское хозяйство?"
result = qa_chain.run(query)
print(result)
