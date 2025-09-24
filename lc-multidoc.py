# Установка необходимых библиотек для работы с разными форматами
# !pip install langchain langchain-community faiss-cpu ollama pypdf2 python-docx openpyxl xlrd > /dev/null

import os
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.llms.ollama import OllamaLLM
from langchain.chains import RetrievalQA

class MultiFormatDocumentLoader:
    """Класс для загрузки документов различных форматов"""
    
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.loaders_config = {
            '.txt': (TextLoader, {'encoding': 'utf-8'}),
            '.pdf': (PyPDFLoader, {}),
            '.docx': (Docx2txtLoader, {}),
            '.doc': (Docx2txtLoader, {}),  # Для старых .doc файлов может потребоваться дополнительная обработка
            '.xlsx': (UnstructuredExcelLoader, {}),
            '.xls': (UnstructuredExcelLoader, {}),
            '.pptx': (UnstructuredPowerPointLoader, {}),
            '.ppt': (UnstructuredPowerPointLoader, {}),
            '.rtf': (TextLoader, {'encoding': 'utf-8'}),  # RTF как текст
        }
    
    def load_documents(self):
        """Загружает все документы из директории с поддержкой различных форматов"""
        all_documents = []
        
        for extension, (loader_class, loader_args) in self.loaders_config.items():
            try:
                pattern = f"*{extension}"
                loader = DirectoryLoader(
                    self.directory_path, 
                    glob=pattern, 
                    loader_cls=loader_class,
                    loader_kwargs=loader_args,
                    silent_errors=True  # Пропускаем файлы с ошибками
                )
                documents = loader.load()
                if documents:
                    print(f"✅ Загружено {len(documents)} файлов формата {extension}")
                    all_documents.extend(documents)
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке файлов {extension}: {e}")
        
        if not all_documents:
            raise ValueError(f"Не найдено поддерживаемых файлов в директории {self.directory_path}")
        
        return all_documents

def get_supported_formats():
    """Возвращает список поддерживаемых форматов"""
    formats = [
        ".txt - Текстовые файлы",
        ".pdf - PDF документы", 
        ".docx/.doc - Документы Word",
        ".xlsx/.xls - Таблицы Excel",
        ".pptx/.ppt - Презентации PowerPoint",
        ".rtf - Rich Text Format"
    ]
    return formats

# Основная программа
def main():
    print("📚 Поддерживаемые форматы файлов:")
    for fmt in get_supported_formats():
        print(f"  • {fmt}")
    print()
    
    # 1. Загружаем документы из директории с поддержкой различных форматов
    docs_directory = "docs/"  # Папка с документами
    
    if not os.path.exists(docs_directory):
        print(f"❌ Директория {docs_directory} не найдена!")
        print("Создайте папку 'docs' и поместите туда ваши документы")
        return
    
    try:
        multi_loader = MultiFormatDocumentLoader(docs_directory)
        documents = multi_loader.load_documents()
        print(f"📂 Всего загружено документов: {len(documents)}")
    except Exception as e:
        print(f"❌ Ошибка при загрузке документов: {e}")
        return

    # 2. Делим документы на чанки
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separator="\n"  # Разделитель для лучшего сохранения структуры
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"✂️ Создано текстовых чанков: {len(docs_chunks)}")

    # 3. Генерируем эмбеддинги с помощью локальной модели через Ollama
    print("🔍 Создание векторных представлений...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = FAISS.from_documents(docs_chunks, embedding=embeddings)
        print("✅ Векторное хранилище создано успешно")
    except Exception as e:
        print(f"❌ Ошибка при создании эмбеддингов: {e}")
        print("Убедитесь, что Ollama запущен и модель 'nomic-embed-text' установлена")
        return

    # 4. Создаем ретривер
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )

    # 5. Строим цепочку с локальной LLM через Ollama
    try:
        llm = OllamaLLM(model="llama3.1", temperature=0.1)  # Уменьшаем температуру для более точных ответов
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True  # Возвращать исходные документы для проверки
        )
        print("✅ Цепочка вопрос-ответ создана успешно")
    except Exception as e:
        print(f"❌ Ошибка при создании цепочки QA: {e}")
        print("Убедитесь, что Ollama запущен и модель 'llama3.1' установлена")
        return

    # 6. Интерактивный режим вопрос-ответ
    print("\n" + "="*60)
    print("🤖 Система готова к работе! Задавайте вопросы по вашим документам.")
    print("Для выхода введите 'quit' или 'exit'")
    print("="*60)
    
    while True:
        try:
            query = input("\n❓ Ваш вопрос: ").strip()
            
            if query.lower() in ['quit', 'exit', 'выход']:
                print("👋 До свидания!")
                break
            
            if not query:
                continue
            
            print("🔎 Поиск информации...")
            result = qa_chain({"query": query})
            
            print("\n💡 Ответ:")
            print(result["result"])
            
            # Показываем источники информации
            if result["source_documents"]:
                print("\n📚 Источники информации:")
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    source = doc.metadata.get('source', 'Неизвестно')
                    page = doc.metadata.get('page', 'N/A')
                    print(f"  {i}. Файл: {os.path.basename(source)} (Страница: {page})")
                    
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
# 📦 Дополнительные инструкции по установке
# Для работы с дополнительными форматами установите:
# Windows & RHEL (общие зависимости):
# bash
# # Основные зависимости для обработки документов
# pip install pypdf2 python-docx openpyxl xlrd unstructured

# # Для улучшенной обработки сложных документов (опционально)
# pip install "unstructured[pdf]" "unstructured[docx]" "unstructured[xlsx]"

# # Дополнительные инструменты для обработки текста
# pip install unstructured-inference
# Дополнительные настройки для RHEL:
# Для работы с некоторыми форматами в RHEL могут потребоваться системные библиотеки:

# bash
# # Установка системных зависимостей в RHEL
# sudo yum install -y poppler-utils  # Для работы с PDF
# sudo yum install -y libjpeg-turbo-devel  # Для обработки изображений в документах
# sudo yum install -y python3-devel  # Для компиляции некоторых Python-пакетов

# # Или для RHEL 8+ с dnf:
# sudo dnf install -y poppler-utils libjpeg-turbo-devel python3-devel
# 🎯 Особенности обновленной программы:
# Поддержка множества форматов: PDF, DOC/DOCX, XLS/XLSX, PPT/PPTX, RTF, TXT

# Обработка ошибок: Программа продолжает работу даже если некоторые файлы повреждены

# Информативное логирование: Показывает прогресс загрузки и обработки

# Интерактивный режим: Возможность задавать multiple questions без перезапуска

# Показ источников: Указывает из каких файлов и страниц была взята информация

# Гибкая настройка: Легко добавить поддержку новых форматов

# 🔧 Проверка установки:
# Создайте тестовый скрипт для проверки зависимостей:

# python
# # test_dependencies.py
# try:
#     import PyPDF2
#     import docx
#     import openpyxl
#     import langchain
#     import ollama
#     print("✅ Все зависимости установлены успешно!")
# except ImportError as e:
#     print(f"❌ Отсутствует зависимость: {e}")
# Теперь ваша RAG-система может работать с документами практически любых популярных форматов!

