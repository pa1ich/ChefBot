# 👨‍🍳 ChefBot — Голосовой кулинарный ассистент с RAG

> 🎯 **Vibe Coding Project**: от идеи до рабочего прототипа за один поток
 
<img width="1084" height="602" alt="image" src="https://github.com/user-attachments/assets/d8ec3f98-5db9-4bb0-8c96-68a025cfda5f" />

---

## 🚀 Что умеет бот

| Функция | Как работает |
|---------|-------------|
| 🎤 Голосовой ввод | Распознавание через **Whisper** (локально) |
| 🔍 Поиск по книгам | Векторный поиск в **Pinecone** (6 кулинарных книг) |
| 🤖 Генерация ответа | Локальная LLM **Qwen2.5:3b** через Ollama |
| 📚 Указание источника | Формат: `[Книга: название, стр. X]` |
| 🔐 Авторизация | Команда `/auth <пароль>` для доступа |

---

## 🧱 Архитектура
<img width="1084" height="602" alt="image" src="https://github.com/user-attachments/assets/aeddfd6e-5dad-48a7-9570-8b49b1f3bba2" />
