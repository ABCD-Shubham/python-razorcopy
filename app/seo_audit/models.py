from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    agency_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    audits = relationship('Audit', back_populates='user')
    schedules = relationship('ScheduledAudit', back_populates='user')

class AuditGroup(Base):
    __tablename__ = 'audit_groups'
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(768), unique=True, nullable=False)
    first_run_at = Column(DateTime)
    last_run_at = Column(DateTime)
    audits = relationship('Audit', back_populates='group')

class Audit(Base):
    __tablename__ = 'audits'
    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey('audit_groups.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    parameters = Column(JSON)
    results = Column(JSON)
    run_at = Column(DateTime, default=datetime.utcnow)
    progress = Column(String(255), default="Queued")
    group = relationship('AuditGroup', back_populates='audits')
    user = relationship('User', back_populates='audits')

class ScheduledAudit(Base):
    __tablename__ = 'scheduled_audits'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    group_id = Column(Integer, ForeignKey('audit_groups.id'), nullable=False)
    cron_expression = Column(String(100), nullable=False)
    next_run_at = Column(DateTime)
    last_run_at = Column(DateTime)
    active = Column(Boolean, default=True)
    user = relationship('User', back_populates='schedules')
    group = relationship('AuditGroup') 