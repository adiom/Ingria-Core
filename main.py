from fastapi import FastAPI, Request, HTTPException, Depends, Header, Form, Cookie
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, APIError
from sqlalchemy.orm import Session
import re
import time
from datetime import datetime, timedelta
from typing import Optional
import os
import uuid
import secrets
from fastapi import status
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Получаем переменные из .env
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

from config.database import get_db, engine
from models.database import Base, Client, Request as DBRequest, UsageStats
from schemas.api import ClientCreate, Request as RequestSchema

# Создаем директории если не существуют
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Создаем таблицы
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ingria Router", description="Autonomous AI Router")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Заголовок для API ключа
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Системная инструкция
system_message = os.getenv("SYSTEM_MESSAGE", "Ты - автономная нейросеть Ingria с собственной эмоциональной системой...")

# Функция для получения текущего пользователя из cookie
async def get_current_user(
    session: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not session:
        return None
    
    try:
        client_id = int(session)
        client = db.query(Client).filter(Client.id == client_id).first()
        return client
    except:
        return None

# API аутентификация
async def get_client_from_api_key(
    api_key: str = Depends(API_KEY_HEADER),
    db: Session = Depends(get_db)
):
    client = db.query(Client).filter(Client.api_key == api_key).first()
    if not client:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return client

@app.post("/v1/clients")
async def create_client(client: ClientCreate, db: Session = Depends(get_db)):
    # Проверяем, существует ли клиент с таким API ключом
    existing = db.query(Client).filter(Client.api_key == client.api_key).first()
    if existing:
        raise HTTPException(status_code=400, detail="Client with this API key already exists")
    
    db_client = Client(**client.dict())
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    
    # Создаем начальную статистику использования
    stats = UsageStats(client_id=db_client.id)
    db.add(stats)
    db.commit()
    
    return db_client

@app.get("/v1/models")
async def get_models(client: Client = Depends(get_client_from_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": "canfly-ingria",
                "object": "model",
                "created": 1677652288,
                "owned_by": "canfly",
                "name": "Ingria",
                "description": "Experimental autonomous AI with emotional simulation",
                "capabilities": {
                    "text": True,
                    "image": False,
                    "emotions": True
                },
                "status": "active"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: Client = Depends(get_client_from_api_key),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        data = await request.json()
        messages = data.get("messages", [])
        model = data.get("model", "google/gemma-3-27b-it:free")
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Добавляем системное сообщение
        with_system = [system_message]
        with_system.extend(messages)
        
        # Создаем клиент OpenRouter с переданным API ключом
        base_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
        api_key = os.getenv("OPENROUTER_API_KEY", "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")
        openrouter_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        
        # Отправляем запрос в OpenRouter
        completion = openrouter_client.chat.completions.create(
            model=model,
            messages=with_system
        )
        
        response_content = completion.choices[0].message.content
        
        # Извлекаем эмоцию
        emotions_match = re.search(r"Эмоция: (\w+)$", response_content, re.MULTILINE)
        emotion = emotions_match.group(1).strip() if emotions_match else "нейтральность"
        
        # Сохраняем запрос в базу данных
        db_request = DBRequest(
            client_id=client.id,
            model=model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            total_tokens=completion.usage.total_tokens,
            request_data=data,
            response_data={
                "content": response_content,
                "emotion": emotion
            },
            emotion=emotion
        )
        db.add(db_request)
        
        # Обновляем статистику использования
        stats = db.query(UsageStats).filter(UsageStats.client_id == client.id).first()
        if not stats:
            stats = UsageStats(client_id=client.id)
            db.add(stats)
        
        response_time = time.time() - start_time
        
        # Обновляем базовую статистику
        stats.total_requests += 1
        stats.total_tokens += completion.usage.total_tokens
        stats.total_prompt_tokens += completion.usage.prompt_tokens
        stats.total_completion_tokens += completion.usage.completion_tokens
        
        # Обновляем статистику времени ответа
        stats.avg_response_time = (
            (stats.avg_response_time * (stats.total_requests - 1) + response_time)
            / stats.total_requests
        )
        if response_time > stats.max_response_time:
            stats.max_response_time = response_time
        if stats.min_response_time == 0 or response_time < stats.min_response_time:
            stats.min_response_time = response_time
        
        # Обновляем статистику эмоций
        if emotion not in stats.emotion_stats:
            stats.emotion_stats[emotion] = 0
        stats.emotion_stats[emotion] += 1
        
        # Обновляем статистику моделей
        if model not in stats.model_stats:
            stats.model_stats[model] = 0
        stats.model_stats[model] += 1
        
        stats.last_updated = datetime.utcnow()
        
        db.commit()
        
        return {
            "id": completion.id,
            "object": "chat.completion",
            "created": completion.created,
            "model": "canfly-ingria",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": completion.choices[0].finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            },
            "emotion": emotion
        }
    
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/v1/analytics/usage/{client_id}")
async def get_client_usage(
    client_id: int,
    client: Client = Depends(get_client_from_api_key),
    db: Session = Depends(get_db)
):
    if client.id != client_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    stats = db.query(UsageStats).filter(UsageStats.client_id == client_id).first()
    if not stats:
        raise HTTPException(status_code=404, detail="Stats not found")
    
    return stats

# Веб-интерфейс

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: Optional[Client] = Depends(get_current_user)):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": user}
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: Optional[Client] = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/dashboard")
    
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "user": None}
    )

@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    api_key: str = Form(...),
    db: Session = Depends(get_db)
):
    client = db.query(Client).filter(Client.api_key == api_key).first()
    if not client:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "user": None, "error": "Неверный API ключ"}
        )
    
    # Создаем ответ с редиректом и устанавливаем cookie
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="session", value=str(client.id), httponly=True)
    
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: Optional[Client] = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/dashboard")
    
    return templates.TemplateResponse(
        "register.html",
        {"request": request, "user": None}
    )

@app.post("/register")
async def register(
    request: Request,
    name: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Проверяем, существует ли клиент с таким именем
        existing = db.query(Client).filter(Client.name == name).first()
        if existing:
            return templates.TemplateResponse(
                "register.html",
                {"request": request, "error": "Проект с таким именем уже существует"}
            )
        
        # Генерируем случайный API ключ
        api_key = secrets.token_urlsafe(32)
        
        # Создаем нового клиента
        client = Client(
            name=name,
            api_key=api_key
        )
        
        db.add(client)
        db.commit()
        db.refresh(client)
        
        # Создаем начальную статистику использования
        stats = UsageStats(client_id=client.id)
        db.add(stats)
        db.commit()
        
        # Создаем ответ с куки
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="session",
            value=str(client.id),
            httponly=True
        )
        
        # Добавляем сообщение об успешной регистрации
        response.headers["X-Success-Message"] = f"Регистрация успешна! Ваш API ключ: {api_key}"
        
        return response
    except Exception as e:
        db.rollback()
        print(f"Ошибка при регистрации: {str(e)}")  # Логируем ошибку
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": f"Ошибка при регистрации: {str(e)}"}
        )

@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie(key="session")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: Optional[Client] = Depends(get_current_user)
):
    if not user:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse(
        "base.html",
        {"request": request, "user": user, "message": "Панель управления в разработке"}
    )

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    user: Optional[Client] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login")
    
    # Получаем историю сообщений
    messages = []
    requests = db.query(DBRequest).filter(DBRequest.client_id == user.id).all()
    for req in requests:
        if "messages" in req.request_data and req.request_data["messages"]:
            user_message = req.request_data["messages"][-1]["content"]
            messages.append({
                "role": "user",
                "content": user_message
            })
            messages.append({
                "role": "assistant",
                "content": req.response_data["content"],
                "emotion": req.emotion
            })
    
    # Получаем статистику
    stats = db.query(UsageStats).filter(UsageStats.client_id == user.id).first()
    if not stats:
        stats = UsageStats(client_id=user.id)
        db.add(stats)
        db.commit()
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": user,
            "messages": messages,
            "stats": {
                "total_messages": len(messages),
                "total_tokens": stats.total_tokens if stats else 0,
                "avg_response_time": stats.avg_response_time if stats else 0.0
            }
        }
    )

@app.post("/chat/send", response_class=HTMLResponse)
async def send_message(
    request: Request,
    message: str = Form(...),
    user: Optional[Client] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login")
    
    # Создаем сообщение для API
    data = {
        "messages": [{"role": "user", "content": message}]
    }
    
    # Преобразуем запрос для использования существующей логики API
    mock_request = Request(scope=request.scope)
    mock_request._json = data
    
    # Отправляем запрос через существующий API endpoint
    try:
        response = await chat_completions(mock_request, user, db)
        
        # Форматируем ответ для шаблона
        return templates.TemplateResponse(
            "_message.html",
            {
                "request": request,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    },
                    {
                        "role": "assistant",
                        "content": response["choices"][0]["message"]["content"],
                        "emotion": response.get("emotion", "нейтральность")
                    }
                ]
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "_message.html",
            {
                "request": request,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    },
                    {
                        "role": "assistant",
                        "content": f"Произошла ошибка: {str(e)}",
                        "emotion": "смущение"
                    }
                ]
            }
        )

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(
    request: Request,
    user: Optional[Client] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse(url="/login")
    
    stats = db.query(UsageStats).filter(UsageStats.client_id == user.id).first()
    
    # Получаем эмоции и их частоту
    emotions = {}
    requests = db.query(DBRequest).filter(DBRequest.client_id == user.id).all()
    for req in requests:
        emotion = req.emotion
        if emotion in emotions:
            emotions[emotion] += 1
        else:
            emotions[emotion] = 1
    
    # Сортируем эмоции по частоте
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request,
            "user": user,
            "message": "Аналитика в разработке",
            "stats": stats,
            "emotions": sorted_emotions
        }
    )

@app.post("/api/auth/register")
async def register(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        data = await request.json()
        name = data.get("name")
        
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        # Проверяем, существует ли клиент с таким именем
        existing = db.query(Client).filter(Client.name == name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Project with this name already exists")
        
        # Генерируем случайный API ключ
        api_key = secrets.token_urlsafe(32)
        
        # Создаем нового клиента
        client = Client(
            name=name,
            api_key=api_key
        )
        
        db.add(client)
        db.commit()
        db.refresh(client)
        
        # Создаем начальную статистику использования
        stats = UsageStats(client_id=client.id)
        db.add(stats)
        db.commit()
        
        return {
            "id": client.id,
            "name": client.name,
            "api_key": api_key
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login")
async def login(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        data = await request.json()
        api_key = data.get("api_key")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        client = db.query(Client).filter(Client.api_key == api_key).first()
        if not client:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return {
            "id": client.id,
            "name": client.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск: uvicorn main:app --host 0.0.0.0 --port 3001