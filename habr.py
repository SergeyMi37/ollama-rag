# https://habr.com/ru/articles/931396/
# Пример 1. RAG с LangChain (векторное хранилище FAISS + OpenAI):

# !pip install langchain faiss-cpu openai tiktoken > /dev/null

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Загружаем документы из директории (например, текстовые файлы)
loader = DirectoryLoader("docs/", glob="*.txt")
documents = loader.load()

# 2. Делим документы на куски (например, по 1000 символов с оверлэпом 100)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_chunks = text_splitter.split_documents(documents)

# 3. Генерируем эмбеддинги для chunk'ов и сохраняем в FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS.from_documents(docs_chunks, embedding=embeddings)

# 4. Создаем ретривер на основе векторного хранилища
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 5. Строим цепочку «Вопрос-Ответ с поиском» (RetrievalQA)
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"),
                                      chain_type="stuff",
                                      retriever=retriever)

# 6. Отвечаем на произвольный вопрос
query = "Что говорится в документах о влиянии климатических изменений на сельское хозяйство?"
result = qa_chain.run(query)
print(result)
# В этом коде мы: загружаем все .txt файлы из папки, режем их на фрагменты по ~1000 символов, получаем для них эмбеддинги через OpenAI, сохраняем в локальный индекс FAISS, затем при запросе извлекаем топ-3 похожих фрагмента и передаем их вместе с вопросом в модель GPT-3.5. Цепочка RetrievalQA (тип stuff) просто «подкладывает» все найденные тексты в промпт. Результат (result) – сгенерированный ответ. В реальном сценарии, вместо печати, можно обернуть это в веб-сервис или чат-интерфейс.

# Пример 2. RAG с LlamaIndex:

# !pip install llama-index > /dev/null

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings import HuggingFaceEmbeddings

# Инициализируем локальную модель эмбеддингов (для примера используем мини-модель)
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
service_context = ServiceContext.from_defaults(embed_model=hf_embed)

# Загружаем документы
documents = SimpleDirectoryReader('docs/').load_data()
# Строим векторный индекс (по умолчанию внутри используется FAISS)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Выполняем запрос
query_engine = index.as_query_engine(response_mode="compact")  # compact: сжатый ответ
response = query_engine.query("Каково влияние климатических изменений на сельское хозяйство?")
print(response.response)
# Здесь LlamaIndex 