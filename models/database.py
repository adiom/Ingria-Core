from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    requests = relationship("Request", back_populates="client")
    usage_stats = relationship("UsageStats", back_populates="client", uselist=False)

class Request(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    model = Column(String)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    request_data = Column(JSON)
    response_data = Column(JSON)
    emotion = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    client = relationship("Client", back_populates="requests")

class UsageStats(Base):
    __tablename__ = "usage_stats"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), unique=True)
    total_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Новые поля для аналитики
    total_prompt_tokens = Column(Integer, default=0)  # Общее количество токенов в запросах
    total_completion_tokens = Column(Integer, default=0)  # Общее количество токенов в ответах
    max_response_time = Column(Float, default=0.0)  # Максимальное время ответа
    min_response_time = Column(Float, default=0.0)  # Минимальное время ответа
    total_errors = Column(Integer, default=0)  # Количество ошибок
    last_error = Column(String, nullable=True)  # Последняя ошибка
    last_error_time = Column(DateTime, nullable=True)  # Время последней ошибки
    emotion_stats = Column(JSON, default=dict)  # Статистика эмоций в формате {"emotion": count}
    model_stats = Column(JSON, default=dict)  # Статистика использования моделей {"model": count}
    
    client = relationship("Client", back_populates="usage_stats") 