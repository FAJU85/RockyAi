"""
Database configuration and models for Rocky AI
SQLAlchemy models and database connection management
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
from apps.api.app.config import get_settings

# Database setup
settings = get_settings()
engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    analyses = relationship("Analysis", back_populates="user")
    datasets = relationship("Dataset", back_populates="user")


class Dataset(Base):
    """Dataset model"""
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)  # csv, json, xlsx, etc.
    columns = Column(JSON, nullable=True)  # Column metadata
    preview_data = Column(JSON, nullable=True)  # First few rows
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    analyses = relationship("Analysis", back_populates="dataset")


class Analysis(Base):
    """Analysis model"""
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    analysis_type = Column(String(100), nullable=False)  # t_test, anova, regression, etc.
    language = Column(String(20), nullable=False)  # python, r
    status = Column(String(50), nullable=False)  # pending, running, completed, failed
    plan = Column(JSON, nullable=True)  # Analysis plan
    code = Column(Text, nullable=True)  # Generated code
    output = Column(Text, nullable=True)  # Execution output
    error = Column(Text, nullable=True)  # Error message if failed
    execution_time = Column(Float, nullable=True)  # Execution time in seconds
    memory_used = Column(Float, nullable=True)  # Memory used in MB
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)
    correlation_id = Column(String(36), nullable=True, index=True)  # For tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    dataset = relationship("Dataset", back_populates="analyses")


class ModelCache(Base):
    """Model response cache"""
    __tablename__ = "model_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    prompt_hash = Column(String(64), nullable=False, index=True)  # SHA256 of prompt
    model_name = Column(String(100), nullable=False)
    response = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)


class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # seconds, MB, count, etc.
    tags = Column(JSON, nullable=True)  # Additional metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


# Database dependency
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database operations
class DatabaseService:
    """Database service for common operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, username: str, email: str, hashed_password: str) -> User:
        """Create new user"""
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def create_dataset(self, name: str, filename: str, file_path: str, 
                      file_size: int, file_type: str, columns: List[Dict],
                      preview_data: List[Dict], user_id: uuid.UUID) -> Dataset:
        """Create new dataset"""
        dataset = Dataset(
            name=name,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            columns=columns,
            preview_data=preview_data,
            user_id=user_id
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset
    
    def get_dataset(self, dataset_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Dataset]:
        """Get dataset by ID and user"""
        return self.db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.user_id == user_id
        ).first()
    
    def list_user_datasets(self, user_id: uuid.UUID) -> List[Dataset]:
        """List user's datasets"""
        return self.db.query(Dataset).filter(Dataset.user_id == user_id).all()
    
    def create_analysis(self, query: str, analysis_type: str, language: str,
                       user_id: Optional[uuid.UUID] = None,
                       dataset_id: Optional[uuid.UUID] = None,
                       correlation_id: Optional[str] = None) -> Analysis:
        """Create new analysis"""
        analysis = Analysis(
            query=query,
            analysis_type=analysis_type,
            language=language,
            status="pending",
            user_id=user_id,
            dataset_id=dataset_id,
            correlation_id=correlation_id
        )
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis
    
    def update_analysis(self, analysis_id: uuid.UUID, **kwargs) -> Optional[Analysis]:
        """Update analysis"""
        analysis = self.db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            for key, value in kwargs.items():
                setattr(analysis, key, value)
            self.db.commit()
            self.db.refresh(analysis)
        return analysis
    
    def get_analysis(self, analysis_id: uuid.UUID) -> Optional[Analysis]:
        """Get analysis by ID"""
        return self.db.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    def list_user_analyses(self, user_id: uuid.UUID, limit: int = 50) -> List[Analysis]:
        """List user's analyses"""
        return self.db.query(Analysis).filter(
            Analysis.user_id == user_id
        ).order_by(Analysis.created_at.desc()).limit(limit).all()
    
    def cache_model_response(self, cache_key: str, prompt_hash: str, 
                           model_name: str, response: str, 
                           tokens_used: Optional[int] = None,
                           ttl_seconds: int = 3600) -> ModelCache:
        """Cache model response"""
        cache_entry = ModelCache(
            cache_key=cache_key,
            prompt_hash=prompt_hash,
            model_name=model_name,
            response=response,
            tokens_used=tokens_used,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds)
        )
        self.db.add(cache_entry)
        self.db.commit()
        self.db.refresh(cache_entry)
        return cache_entry
    
    def get_cached_response(self, cache_key: str) -> Optional[ModelCache]:
        """Get cached model response"""
        return self.db.query(ModelCache).filter(
            ModelCache.cache_key == cache_key,
            ModelCache.expires_at > datetime.utcnow()
        ).first()
    
    def record_metric(self, metric_name: str, metric_value: float, 
                     metric_unit: Optional[str] = None, tags: Optional[Dict] = None):
        """Record system metric"""
        metric = SystemMetrics(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            tags=tags or {}
        )
        self.db.add(metric)
        self.db.commit()


# Initialize database tables
def init_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


# Health check
def check_database_health() -> bool:
    """Check database connectivity"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception:
        return False
