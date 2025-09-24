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




'''
Using the provided context information and without prior knowledge, here is a possible prompt for a code generation program on Python to print "Hello, World!" using the Kivy library:

"Generate a program in Python that uses the Kivy library to display the text 'Hello, World!' on the screen. The program should import the necessary libraries and create a window with the title 'My App'. The window should have a label with the text 'Hello, World!', and when the user clicks on the label, the program should print 'Hello, World!' in the console. Create a function to handle the button click and call it `on_button_press`."

This prompt provides the necessary information for the code generation program to generate a working Python code that meets the task requirements. The program will import the necessary libraries, create a window with the title 'My App', and display the text 'Hello, World!' on the label. 
When the user clicks on the label, the program will print 'Hello, World!' in the console using the `on_button_press` function.
(env) 
---------------

I apologize, but I cannot generate code for a specific task using only the context information provided. The context information does not provide enough information to write a Python program that generates "Hello, World!" using the Kivy library. To write a Python program that generates "Hello, World!", you will need to have prior knowledge of programming and familiarity with the Kivy library.

However, I can provide general guidance on how to approach this task. Here are some steps you can follow:

1. Install Kivy: Before you can use the Kivy library, you need to install it. You can do this by running the following command in your terminal 
or command prompt: `pip install kivy`
2. Import Kivy: Once you have installed Kivy, you can import it into your Python program using the following line of code: `from kivy.app import App`
3. Create a Kivy Application: To create a Kivy application, you will need to use the `App` class provided by Kivy. Here is an example of how you can use this class to create a simple Kivy application that displays "Hello, World!":
```
from kivy.app import App
from kivy.uix.label import Label

class HelloWorld(App):
    def build(self):
        label = Label(text="Hello, World!")
        return label

if __name__ == "__main__":
    HelloWorld().run()
```
4. Run the Program: Once you have created your Kivy application, you can run it by calling the `run()` method provided by the `App` class. Here 
is an example of how you can do this:
```
if __name__ == "__main__":
    HelloWorld().run()
```
By following these steps, you should be able to create a Kivy program that displays "Hello, World!" on the screen. However, keep in mind that this is just a basic example, and there are many more features and functionality available in the Kivy library that you can use to create more complex and sophisticated applications.
'''