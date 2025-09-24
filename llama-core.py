# Пример 2. RAG с LlamaIndex:
# !pip install llama-index > /dev/null

# pip uninstall llama-index
# pip install llama-index-core
# pip install llama-index-llms-ollama
# pip install llama-index-embeddings-ollama

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.response_synthesizers import ResponseMode

# Настройка Ollama
OLLAMA_URL = "http://127.0.0.1:11434"
print('----0')
# Инициализируем embedding-модель из Ollama
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_URL,
    timeout=120.0  # Таймаут для embedding-модели
)
print('----1')
llm = Ollama(
    model="llama2",
    base_url=OLLAMA_URL,
    request_timeout=120.0
)
print('----2')
Settings.llm = llm  # Например, ваша модель Ollama
Settings.embed_model = embed_model  # Например, ваша модель эмбеддингов Ollama
Settings.chunk_size = 1000
Settings.chunk_overlap = 100
print('----3')

# Загружаем документы
documents = SimpleDirectoryReader('docs/').load_data()
print('----4',documents)

# Строим векторный индекс
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,  # Передача LLM
    embed_model=embed_model  # Передача модели эмбеддингов
)
print('----5')
# Выполняем запрос
query_engine = index.as_query_engine(
    response_mode=ResponseMode.COMPACT,
    timeout=600.0,  # 10 минут
    similarity_top_k=3,  # Количество похожих фрагментов
    streaming=False,
    llm_kwargs={
        "temperature": 0.1,
        "max_tokens": 2000
    }
    )
print('----6')
# prompt  = "Используя исходные документы RAG, напиши промпт для генерации кода на Python для решения следующей задачи:" \
# " Напи программу, которая будет выводить на экран текст 'Hello, World!' использую библитеку kivy."
prompt  = "Используя исходные документы, выбери номер темы, который больше все подходит для текста: 'Отсутствует водоснабжение'" 

response = query_engine.query(prompt)

print('----7')
print(response.response)

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