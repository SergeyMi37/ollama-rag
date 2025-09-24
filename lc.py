from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Укажите адрес удалённого сервера Ollama
OLLAMA_URL = "http://127.0.0.1:11434"
print('----0')
# 1. Загружаем документы из директории
loader = DirectoryLoader("docs/", glob="*.txt")
print('----1')
documents = loader.load()
print('----2')
# 2. Делим документы на чанки
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
print('----3')
docs_chunks = text_splitter.split_documents(documents)
print('----4')
# 3. Генерируем эмбеддинги с помощью локальной модели через Ollama
# Убедитесь, что модель для эмбеддингов (например, nomic-embed-text) запущена в Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)  # Популярная модель для эмбеддингов
print('----5')
vector_store = FAISS.from_documents(docs_chunks, embedding=embeddings)
print('----6')
# 4. Создаем ретривер
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print('----7')
# 5. Строим цепочку с локальной LLM через Ollama
# Убедитесь, что модель для генерации (например, llama3.1) запущена в Ollama
llm = OllamaLLM(model="llama3.1", base_url=OLLAMA_URL, temperature=0)  # Укажите желаемую модель, например, "llama3.1", "qwen2.5:7b"
print('----8')
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type="stuff",
                                      retriever=retriever)

# 6. Задаем вопрос
query = "Что говорится в документах о влиянии климатических изменений на сельское хозяйство?"
print('----9')
result = qa_chain.run(query)
print('----10',result)