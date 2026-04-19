# 🗄️ Базы данных в проекте ChefBot

> **Критерий 9:** Наличие базы данных

---

## 📊 Используемые базы данных

### 1. Pinecone (Векторная БД)

**Тип:** Vector Database  
**Назначение:** Хранение эмбеддингов кулинарных книг для семантического поиска

#### Конфигурация:
```python
# Индексация книг
index_name = "chefbot"
namespace = "cheff"
embedding_model = "intfloat/multilingual-e5-large"

# Объем данных:
- 6 кулинарных книг (PDF)
- ~500-1000 чанков текста
- Размерность векторов: 1024

Что хранится:

Поле	Тип	Описание
`id`	string	Уникальный ID чанка
`values`	float[]	Векторное представление (1024 измерения)
`metadata.text`	string	Текст рецепта/инструкции
`metadata.source`	string	Название книги (PDF файл)
`metadata.page`	int	Номер страницы в книге
`metadata.chunk_index`	int	Индекс чанка
