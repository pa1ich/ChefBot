# 🤖 RAG-Assistant Workflow Configuration (Flowise Equivalent)

> **Проект:** ChefBot  
> **Тип:** Retrieval-Augmented Generation (RAG) Pipeline  
> **Реализация:** Python (`main.py` / `PineconeRAG` class)  
> **Эквивалент:** Flowise AI Agent Configuration

---

## 📊 Визуальная схема потока 
Ниже представлена структура агента. В Flowise это выглядело бы как соединение следующих нод (Nodes):

```mermaid
graph LR
    A[🎤 Input / Chat Input] --> B(🔍 Embeddings)
    B --> C[🗄️ Pinecone Vector Store]
    C --> D{📥 Retriever}
    D --> E[🧠 Ollama LLM]
    E --> F[📝 Prompt Template]
    F --> G[💬 Output Parser]

#
🔄 Логика работы Агента
Input: Пользователь отправляет текст или голос (распознанный Whisper).
Embed: Запрос преобразуется в вектор (500+ dimensions).
Retrieve: Pinecone находит 5 наиболее похожих чанков текста из книг.
Context Assembly: Текст чанков + Метаданные (Название книги, Страница) собираются в промпт.
Generate: Ollama (Qwen2.5) генерирует ответ на основе контекста.
Output: Ответ возвращается пользователю с указанием источника.
#
В проекте реализован полный цикл Retrieval-Augmented Generation (RAG):
1️⃣ **Retrieval (Поиск)**: 
   Пользовательский запрос преобразуется в эмбеддинг через `multilingual-e5-large` 
   и отправляется в векторную базу Pinecone. Система возвращает топ-5 наиболее 
   релевантных фрагментов из 6 кулинарных книг.
#
2️⃣ **Augmentation (Дополнение контекста)**: 
   Каждый найденный фрагмент снабжается метаданными (название книги, номер страницы). 
   Фрагменты собираются в единый блок `Контекст:` и внедряются в системный промпт.
#
3️⃣ **Generation (Генерация)**:
   Локальная LLM `qwen2.5:3b` (через Ollama) получает промпт с жёсткой инструкцией: 
   «Используй только информацию из контекста. Укажи источник в формате [Книга: ..., стр. X]». 
   Это исключает галлюцинации и гарантирует проверяемость ответа.
📍 **Код находится в файле `main.py`, класс `PineconeRAG`, методы `search()` и `ask()`.**
Код, подтверждающий RAG-архитектуру

```
# 1. RETRIEVAL (Поиск релевантных данных в векторной БД) 
def search(self, query: str, top_k: int = 5):
    # Преобразуем текстовый запрос в векторное представление
    query_emb = self.embedder.encode(query).tolist()
    
    # Выполняем семантический поиск в Pinecone
    results = self.index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=self.namespace,
        include_metadata=True
    )
    
    # Извлекаем текст + метаданные (источник, страница)
    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", ""), 
            "page": match.metadata.get("page", 0),       
            "score": match.score
        })
    return chunks

# 2. AUGMENTATION (Дополнение промпта контекстом)
async def ask(self, query: str) -> str:
    # Получаем релевантные чанки из базы знаний
    chunks = self.search(query, top_k=5)
    if not chunks:
        return "😕 В книгах не нашлось рецептов..."

    # Формируем контекст с явным указанием источников
    primary_context = "\n\n---\n\n".join(
        f"[Книга: {chunk['source']}, страница {chunk['page']}]\n{chunk['text']}"
        for chunk in chunks[:5]
    )

    # Создаём промпт с инструкцией опираться только на контекст
    primary_prompt = f"""Ты — опытный шеф-повар, помогающий по старинным кулинарным книгам. 
У пользователя есть ингредиенты: {query}. 
Ниже приведены отрывки из книг, наиболее похожие на запрос. 
Выбери самый подходящий рецепт из контекста и опиши его пошагово.
Обязательно укажи, из какой книги и с какой страницы взят рецепт (формат: [Книга: название, стр. X]).
Не придумывай новые рецепты, используй только информацию из контекста.

Контекст:
{primary_context}

Рецепт:
# 3. GENERATION (Генерация ответа через LLM) 
    main_answer = await self._call_ollama(primary_prompt)
    return main_answer
```

```
# RAG класс для Pinecone (поддержка нескольких книг)
class PineconeRAG:
    def __init__(self, api_key: str, index_name: str, namespace: str, host: str, pdf_paths: list):
        self.index_name = index_name
        self.namespace = namespace
        self.pdf_paths = pdf_paths
        self.ollama_base_url = "http://localhost:11434"
        self.llm_model = "qwen2.5:3b"  # ⚡ Рекомендуемая быстрая и качественная модель
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index = self.pc.Index(host=host)

        # Загружаем модель эмбеддингов
        print("🔄 Загрузка модели эмбеддингов (intfloat/multilingual-e5-large)...")
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
        print("✅ Модель эмбеддингов загружена")

        # Индексируем все книги
        self._load_and_index_all_pdfs()

    def _extract_text_from_pdf(self, pdf_path: str) -> list[tuple[int, str]]:
        reader = PdfReader(pdf_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((i+1, text))
        return pages

    def _split_page_into_chunks(self, page_num: int, page_text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[tuple[str, int]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(page_text)
        return [(chunk, page_num) for chunk in chunks]

    def _generate_id(self, text: str, book_name: str, page_num: int, chunk_index: int) -> str:
        unique = f"{book_name}_{page_num}_{chunk_index}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        return hashlib.md5(unique.encode()).hexdigest()[:16]

    def _load_and_index_all_pdfs(self):
        # Очищаем namespace перед загрузкой
        try:
            print(f"⚠️ Очищаем namespace '{self.namespace}' от старых векторов...")
            self.index.delete(delete_all=True, namespace=self.namespace)
            print("✅ Старые векторы удалены.")
        except Exception as e:
            print(f"Ошибка при удалении векторов (возможно, их не было): {e}")

        all_chunks = []  # (текст, имя_книги, номер_страницы)

        for pdf_path in self.pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️ Книга не найдена: {pdf_path}, пропускаем...")
                continue
            print(f"📖 Загрузка книги: {pdf_path}")
            pages = self._extract_text_from_pdf(pdf_path)
            book_name = os.path.basename(pdf_path)
            for page_num, page_text in pages:
                chunks_with_page = self._split_page_into_chunks(page_num, page_text)
                for chunk, page in chunks_with_page:
                    all_chunks.append((chunk, book_name, page))
            print(f"   Извлечено страниц: {len(pages)}")

        print(f"📄 Всего получено {len(all_chunks)} чанков из всех книг")

        if not all_chunks:
            print("❌ Нет данных для индексации.")
            return

        # Создаём эмбеддинги для всех чанков
        print("🔄 Создание эмбеддингов...")
        texts = [chunk[0] for chunk in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # Подготавливаем записи для Pinecone
        vectors = []
        for i, ((text, book_name, page_num), emb) in enumerate(zip(all_chunks, embeddings)):
            vectors.append({
                "id": self._generate_id(text, book_name, page_num, i),
                "values": emb.tolist(),
                "metadata": {
                    "text": text,
                    "source": book_name,
                    "page": page_num,
                    "chunk_index": i
                }
            })
```
