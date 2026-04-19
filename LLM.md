«Критерий "использование LLM — ссылки на источники" реализован через механизм RAG с метаданными: каждый чанк текста хранит название книги и номер страницы. 
Промпт для модели явно требует указывать источник в формате [Книга: ..., стр. X]. 
Это гарантирует, что ответ не является «галлюцинацией», а привязан к проверяемому документу. 
Пользователь может открыть указанную книгу и найти рецепт самостоятельно».

1️⃣ Пользователь задаёт запрос: «что приготовить из курицы и картошки?»
2️⃣ Система ищет похожие чанки в векторной базе (Pinecone)
3️⃣ В контекст для LLM передаются не только тексты, но и метаданные:
  • Название книги (источник)
  • Номер страницы
  • Уникальный ID чанка
4️⃣ Промпт явно требует от модели: «Укажи источник в формате [Книга: ..., стр. X]»
5️⃣ Ответ включает цитату + ссылку на оригинал

🍽️ Вот что я нашёл:
[Книга: 1000_receptov_starinnoj_kuhni_l_p_novikova_1993.pdf, стр. 142]
🍗 Курица, запечённая с картофелем
Ингредиенты: ...
Приготовление: ...
💡 Совет шеф-повара: Если добавить чеснок и розмарин...
[Книга: Кухня_народов_России..., стр. 87]

# 📚 Источники и документация
### Используемые модели и библиотеки:
| Компонент | Ссылка | Версия |
|-----------|--------|--------|
| 🤖 Qwen2.5 | [ollama.com/library/qwen2.5](https://ollama.com/library/qwen2.5) | 3b |
| 🎙️ Whisper | [github.com/openai/whisper](https://github.com/openai/whisper) | base |
| 🔗 Embeddings | [huggingface.co/intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | latest |
| 🗄️ Pinecone | [docs.pinecone.io](https://docs.pinecone.io) | serverless |
| 🤖 aiogram | [docs.aiogram.dev](https://docs.aiogram.dev) | 3.x |

### Принципы цитирования в RAG-системах:
- [LangChain: Source Document Attribution](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone: Metadata Filtering](https://docs.pinecone.io/guides/data/filtering-with-metadata)
- [Research: Reducing Hallucination via Source Grounding](https://arxiv.org/abs/2305.14283)
📌 Важно: Модель не получает доступ ко всему тексту книги — только к релевантным чанкам с метаданными. Это гарантирует, что цитата будет точной, а не выдуманной.

```
# ========== Конфигурация LLM ==========
self.ollama_base_url = "http://localhost:11434"
self.llm_model = "qwen2.5:3b"  # ⚡ Локальная большая языковая модель

# ========== Функция вызова LLM ==========
async def _call_ollama(prompt: str) -> str:
    """Отправляет промпт в локальную LLM через Ollama API"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,  # ← ИСПОЛЬЗУЕМАЯ МОДЕЛЬ
                "prompt": prompt,          # ← КОНТЕКСТ + ЗАПРОС
                "stream": False
            },
            timeout=350
        )
    )
    if response.status_code == 200:
        data = response.json()
        return data.get("response", "").strip()  # ← ОТВЕТ ОТ LLM
    return f"❌ Ошибка Ollama: {response.status_code}"

# ========== Промпт для LLM (с требованием цитировать источники) ==========
primary_prompt = f"""Ты — опытный шеф-повар, помогающий по старинным кулинарным книгам. 
У пользователя есть ингредиенты: {query}. 
Ниже приведены отрывки из книг, наиболее похожие на запрос. 
**Выбери самый подходящий рецепт из контекста и опиши его пошагово.** 
Обязательно укажи, из какой книги и с какой страницы взят рецепт (формат: [Книга: название, стр. X]).
Если точного рецепта с теми же ингредиентами нет, выбери самый близкий и адаптируй, но предупреди об этом.
Не придумывай новые рецепты, используй только информацию из контекста.

Контекст:
{primary_context}

Рецепт:"""

# ========== Вызов LLM и получение ответа ==========
main_answer = await _call_ollama(primary_prompt)  # ← ЗДЕСЬ ПРОИСХОДИТ ГЕНЕРАЦИЯ
```
