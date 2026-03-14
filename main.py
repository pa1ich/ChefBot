import os
import asyncio
import tempfile
import requests
from pathlib import Path
from aiogram.exceptions import TelegramNetworkError

import whisper
import librosa
import soundfile as sf
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# ========== RAG IMPORTS ==========
import pinecone
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import time

# ========== Загрузка .env ==========
env_path = Path(__file__).parent / 'tokens.env'
print(f"🔍 Загрузка .env из: {env_path}")
print(f"📁 Файл существует: {env_path.exists()}")

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ .env загружен успешно")
else:
    print("❌ Файл .env не найден в директории:", Path(__file__).parent)
    print("📋 Создайте файл tokens.env в папке:", Path(__file__).parent)

BOT_TOKEN = os.getenv("BOT_TOKEN")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не найден в .env файле!")
print(f"✅ BOT_TOKEN загружен: {BOT_TOKEN[:10]}...")

# ========== Параметры Pinecone ==========
PINECONE_INDEX_NAME = "chefbot"
PINECONE_NAMESPACE = "cheff"
PINECONE_HOST = "https://f-r3aeqyf.svc.aped-4627-b74a.pinecone.io"

# Список всех книг (добавлены новые)
BOOK_PATHS = [
    Path(__file__).parent / "1000_receptov_starinnoj_kuhni_l_p_novikova_1993.pdf",
    Path(__file__).parent / "bookofNinaBorisova.pdf",
    Path(__file__).parent / "Из_Турции_с_любовью_Турецкая_кухня_от_Стамбула_до_Мардина.pdf",
    Path(__file__).parent / "Prosto_zavtrak_Retcepty_dlia_teh_kto_liubit_vkusno_zhit.pdf",
    Path(__file__).parent / "Кухня_народов_России_Живое_тепло_кулинарных_традиций.pdf",
    Path(__file__).parent / "V_duhovke_Miaso_ryba_ovoschi_i_deserty.pdf",
]

# ========== Инициализация бота ==========
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
authorized_users = set()

# ========== Модель Whisper ==========
print("🔄 Загрузка модели Whisper...")
try:
    whisper_model = whisper.load_model("base")
    print("✅ Модель Whisper загружена!")
except Exception as e:
    print(f"❌ Ошибка загрузки Whisper: {e}")
    whisper_model = None

# ========== RAG класс для Pinecone (поддержка нескольких книг) ==========
class PineconeRAG:
    def __init__(self, api_key: str, index_name: str, namespace: str, host: str, pdf_paths: list):
        self.index_name = index_name
        self.namespace = namespace
        self.pdf_paths = pdf_paths
        self.ollama_base_url = "http://localhost:11434"
        self.llm_model = "qwen2.5:3b"  # ⚡ Рекомендуемая быстрая и качественная модель

        # Подключаемся к Pinecone
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index = self.pc.Index(host=host)

        # Загружаем модель эмбеддингов
        print("🔄 Загрузка модели эмбеддингов (intfloat/multilingual-e5-large)...")
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
        print("✅ Модель эмбеддингов загружена")

        # Индексируем все книги
        self._load_and_index_all_pdfs()

    def _extract_text_from_pdf(self, pdf_path: str) -> list[tuple[int, str]]:
        """Извлекает текст из одного PDF постранично. Возвращает список (номер_страницы, текст_страницы)."""
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
        """Загружает все PDF, разбивает на чанки и индексирует в Pinecone."""
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

        # Загружаем батчами по 100
        print(f"⬆️ Загрузка в Pinecone (namespace: {self.namespace})...")
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            print(f"  Загружено {min(i+batch_size, len(vectors))}/{len(vectors)}")

        print(f"✅ Индексация завершена: {len(vectors)} чанков")

    def search(self, query: str, top_k: int = 5):
        query_emb = self.embedder.encode(query).tolist()
        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )
        chunks = []
        for match in results.matches:
            chunks.append({
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "source": match.metadata.get("source", ""),
                "page": match.metadata.get("page", 0)
            })
        return chunks

    async def ask(self, query: str) -> str:
        try:
            chunks = self.search(query, top_k=12)  # больше кандидатов

            print(f"\n🔎 Запрос: {query}")
            print(f"📊 Найдено чанков: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"  Чанк {i+1} | score={chunk['score']:.4f} | книга: {chunk['source']}, стр. {chunk['page']} | текст: {chunk['text'][:100]}...")

            if not chunks:
                return "😕 В книгах не нашлось рецептов, близких к вашим ингредиентам."

            primary_chunks = chunks[:5]          # больше контекста для основного рецепта
            secondary_chunks = chunks[5:7] if len(chunks) > 5 else []  # не более 2 для рекомендации

            async def _call_ollama(prompt: str) -> str:
                loop = asyncio.get_event_loop()
                start = time.time()
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{self.ollama_base_url}/api/generate",
                        json={"model": self.llm_model, "prompt": prompt, "stream": False},
                        timeout=350
                    )
                )
                elapsed = time.time() - start
                print(f"⏱️ Ollama ответил за {elapsed:.2f} сек")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                else:
                    return f"❌ Ошибка Ollama: {response.status_code}"

            # Основной рецепт – усиленный промпт
            primary_context = "\n\n---\n\n".join(
                f"[Книга: {chunk['source']}, страница {chunk['page']}]\n{chunk['text']}"
                for chunk in primary_chunks
            )
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

            main_answer = await _call_ollama(primary_prompt)
            if not main_answer:
                main_answer = "Не удалось сформировать рецепт."

            # Рекомендация – короткая и по делу
            recommendation = ""
            if secondary_chunks:
                secondary_context = "\n\n---\n\n".join(
                    f"[Книга: {chunk['source']}, страница {chunk['page']}]\n{chunk['text']}"
                    for chunk in secondary_chunks
                )
                secondary_prompt = f"""У пользователя ингредиенты: {query}. 
Вот ещё похожие рецепты из книг. Если какой-то из них требует всего 1-2 дополнительных ингредиента, которые легко добавить, предложи его как рекомендацию. 
Если ничего не подходит, напиши ""

Контекст:
{secondary_context}

Краткая рекомендация:"""
                rec = await _call_ollama(secondary_prompt)
                if rec and rec.strip():
                    recommendation = f"\n\n💡 <b>Совет шеф-повара:</b> {rec}"

            return f"{main_answer}{recommendation}"

        except Exception as e:
            print(f"RAG error: {e}")
            return f"❌ Ошибка при обработке запроса: {e}"

# ========== Инициализация RAG ==========
rag = None

async def init_rag():
    global rag
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY не найден в .env")
        return

    existing_books = [str(p) for p in BOOK_PATHS if p.exists()]
    if not existing_books:
        print("❌ Ни одна из книг не найдена в папке проекта.")
        return

    rag = PineconeRAG(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
        host=PINECONE_HOST,
        pdf_paths=existing_books
    )
    print("✅ Pinecone RAG готов")

async def query_rag(query: str) -> str:
    global rag
    if rag is None:
        return "❌ RAG не инициализирован. Проверь API ключ и наличие книг."
    return await rag.ask(query)

# ------------------- Команды -------------------
@dp.message(Command("start"))
async def cmd_start(message: Message):
    user_id = message.from_user.id
    if user_id in authorized_users:
        await message.answer(
            "👋 <b>ChefBot</b> — твой голосовой кулинарный помощник\n\n"
            "📌 <b>Что я умею:</b>\n"
            "• Принимать голосовые сообщения с продуктами\n"
            "• Находить рецепты из кулинарных книг\n"
            "• Давать пошаговые инструкции\n\n"
            "🎤 Просто отправь голосовое сообщение, например:\n"
            "<i>«У меня есть курица, картошка и лук, что приготовить?»</i>"
        )
    else:
        await message.answer(
            "🔐 <b>Доступ ограничен</b>\n\n"
            "Для использования бота необходимо авторизоваться.\n"
            "Отправьте команду: /auth <пароль>",
            parse_mode=None
        )

@dp.message(Command("auth"))
async def cmd_auth(message: Message):
    user_id = message.from_user.id
    args = message.text.split(maxsplit=1)

    if len(args) < 2:
        await message.answer("❌ Использование: /auth <пароль>", parse_mode=None)
        return

    password = args[1].strip()
    if password == AUTH_PASSWORD:
        authorized_users.add(user_id)
        await message.answer(
            "✅ <b>Авторизация успешна!</b>\n\n"
            "👨‍🍳 Добро пожаловать в ChefBot!\n\n"
            "📌 <b>Что я умею:</b>\n"
            "• Принимать голосовые сообщения с продуктами\n"
            "• Находить рецепты из кулинарных книг\n"
            "• Давать пошаговые инструкции\n\n"
            "🎤 Отправь мне голосовое сообщение, например:\n"
            "<i>«Что приготовить из картошки и яиц?»</i>"
        )
    else:
        await message.answer("❌ Неверный пароль.")

@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📚 <b>Помощь по ChefBot</b>\n\n"
        "🔹 /start - начать работу\n"
        "🔹 /auth <пароль> - авторизация\n"
        "🔹 /help - это сообщение\n\n"
        "🎤 <b>Голосовые сообщения:</b>\n"
        "Просто отправьте голосовое сообщение с перечнем продуктов,\n"
        "и я найду подходящий рецепт!"
    )

# ------------------- Обработка голосовых сообщений -------------------
@dp.message(lambda message: message.voice is not None)
async def handle_voice(message: Message):
    user_id = message.from_user.id

    if user_id not in authorized_users:
        await message.answer(
            "🔐 Сначала авторизуйтесь: /auth <пароль>",
            parse_mode=None
        )
        return

    if not whisper_model:
        await message.answer("❌ Модель распознавания речи не загружена. Попробуйте позже.")
        return

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    status_msg = await message.answer("🎤 Обрабатываю голосовое сообщение...")

    tmp_ogg_path = None
    wav_path = None

    try:
        file_id = message.voice.file_id
        file = await bot.get_file(file_id)
        voice_ogg = await bot.download_file(file.file_path)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
            tmp_ogg.write(voice_ogg.read())
            tmp_ogg_path = tmp_ogg.name

        await status_msg.edit_text("🔄 Конвертирую аудио...")

        wav_path = tmp_ogg_path.replace(".ogg", ".wav")
        audio, sr = librosa.load(tmp_ogg_path, sr=16000, mono=True)
        sf.write(wav_path, audio, sr)

        await status_msg.edit_text("🤖 Распознаю речь...")

        result = whisper_model.transcribe(wav_path, language="ru", task="transcribe")
        recognized_text = result["text"].strip()

        if not recognized_text:
            await status_msg.edit_text("😕 Не удалось распознать речь. Попробуйте ещё раз.")
            return

        await status_msg.edit_text(f"📝 <b>Распознано:</b>\n<i>{recognized_text}</i>")

        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        await status_msg.edit_text("🍳 Ищу подходящие рецепты...")

        response_text = await query_rag(recognized_text)

        await message.answer(
            f"🍽️ <b>Вот что я нашёл:</b>\n\n{response_text}",
            parse_mode=ParseMode.HTML
        )

        await status_msg.delete()

    except Exception as e:
        # Отключаем parse_mode при выводе ошибки
        await status_msg.edit_text(f"❌ Произошла ошибка: {str(e)}", parse_mode=None)
        print(f"Error in handle_voice: {e}")

    finally:
        for path in [tmp_ogg_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

# ------------------- Обработка текстовых сообщений -------------------
@dp.message()
async def handle_text(message: Message):
    user_id = message.from_user.id

    if user_id not in authorized_users:
        await message.answer(
            "🔐 Сначала авторизуйтесь: /auth <пароль>",
            parse_mode=None
        )
        return

    if message.text.startswith('/'):
        return

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    response = await query_rag(message.text)
    await message.answer(f"🍽️ <b>Вот что я нашёл:</b>\n\n{response}")

# ------------------- Запуск бота -------------------
async def main():
    print("🚀 Запуск бота...")
    print(f"🤖 Bot token: {BOT_TOKEN[:10]}...")
    print(f"🔑 Auth password: {'установлен' if AUTH_PASSWORD else 'не установлен'}")
    await init_rag()
    print("✅ Бот готов к работе!")

    while True:
        try:
            await dp.start_polling(bot, polling_timeout=30, relaxation=2)
        except TelegramNetworkError as e:
            print(f"⚠️ Ошибка сети: {e}. Переподключение через 5 секунд...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())