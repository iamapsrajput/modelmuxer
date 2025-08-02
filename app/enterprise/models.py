# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
# ModelMuxer Enterprise Multi-Tenancy Models
# Database models for organization management and multi-tenancy

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Numeric, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import uuid

Base = declarative_base()


class PlanType(str, Enum):
    """Subscription plan types."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class OrganizationStatus(str, Enum):
    """Organization status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"


class UserRole(str, Enum):
    """User roles within organizations."""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class AuditAction(str, Enum):
    """Audit log action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    API_CALL = "api_call"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_ADDED = "provider_added"
    PROVIDER_REMOVED = "provider_removed"
    USER_INVITED = "user_invited"
    USER_REMOVED = "user_removed"
    SETTINGS_CHANGED = "settings_changed"


class Organization(Base):
    """Organization model for multi-tenancy."""
    __tablename__ = "organizations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # Subscription details
    plan_type = Column(SQLEnum(PlanType), nullable=False, default=PlanType.FREE)
    status = Column(SQLEnum(OrganizationStatus), nullable=False, default=OrganizationStatus.ACTIVE)
    trial_ends_at = Column(DateTime)
    subscription_id = Column(String(255))  # External subscription ID
    
    # Limits and quotas
    monthly_request_limit = Column(Integer, default=1000)
    monthly_cost_limit = Column(Numeric(10, 4), default=10.0)
    max_users = Column(Integer, default=5)
    max_api_keys = Column(Integer, default=10)
    
    # Settings
    settings = Column(JSON, default=dict)
    custom_branding = Column(JSON, default=dict)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(36))
    
    # Relationships
    users = relationship("OrganizationUser", back_populates="organization", cascade="all, delete-orphan")
    budgets = relationship("OrganizationBudget", back_populates="organization", cascade="all, delete-orphan")
    providers = relationship("OrganizationProvider", back_populates="organization", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("OrganizationAPIKey", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime)
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    password_changed_at = Column(DateTime, default=func.now())
    
    # Preferences
    preferences = Column(JSON, default=dict)
    timezone = Column(String(50), default="UTC")
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    organizations = relationship("OrganizationUser", back_populates="user")


class OrganizationUser(Base):
    """Many-to-many relationship between organizations and users."""
    __tablename__ = "organization_users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    permissions = Column(JSON, default=list)  # Additional granular permissions
    
    # Status
    is_active = Column(Boolean, default=True)
    invited_at = Column(DateTime, default=func.now())
    joined_at = Column(DateTime)
    invited_by = Column(String(36), ForeignKey("users.id"))
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    user = relationship("User", back_populates="organizations", foreign_keys=[user_id])
    inviter = relationship("User", foreign_keys=[invited_by])


class OrganizationBudget(Base):
    """Organization-level budget management."""
    __tablename__ = "organization_budgets"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    
    # Budget configuration
    budget_type = Column(String(20), nullable=False)  # daily, weekly, monthly, yearly
    budget_limit = Column(Numeric(10, 4), nullable=False)
    provider = Column(String(50))  # Optional: provider-specific budget
    model = Column(String(100))    # Optional: model-specific budget
    
    # Alert configuration
    alert_thresholds = Column(JSON, default=lambda: [50.0, 80.0, 95.0])
    alert_emails = Column(JSON, default=list)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(36), ForeignKey("users.id"))
    
    # Relationships
    organization = relationship("Organization", back_populates="budgets")
    creator = relationship("User", foreign_keys=[created_by])


class OrganizationProvider(Base):
    """Organization-specific provider configurations."""
    __tablename__ = "organization_providers"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    
    # Provider details
    provider_name = Column(String(50), nullable=False)
    provider_config = Column(JSON, nullable=False)  # API keys, endpoints, etc.
    custom_models = Column(JSON, default=list)      # Custom model configurations
    
    # Routing preferences
    routing_weight = Column(Integer, default=100)   # Weight for routing decisions
    is_enabled = Column(Boolean, default=True)
    is_fallback = Column(Boolean, default=False)
    
    # Rate limiting
    requests_per_minute = Column(Integer, default=60)
    requests_per_hour = Column(Integer, default=1000)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(36), ForeignKey("users.id"))
    
    # Relationships
    organization = relationship("Organization", back_populates="providers")
    creator = relationship("User", foreign_keys=[created_by])


class OrganizationAPIKey(Base):
    """Organization API keys with scoped permissions."""
    __tablename__ = "organization_api_keys"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    
    # Key details
    key_hash = Column(String(255), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Permissions and scopes
    scopes = Column(JSON, default=list)
    permissions = Column(JSON, default=list)
    
    # Usage limits
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)
    monthly_cost_limit = Column(Numeric(10, 4))
    
    # Status and expiration
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(36), ForeignKey("users.id"))
    
    # Relationships
    organization = relationship("Organization", back_populates="api_keys")
    creator = relationship("User", foreign_keys=[created_by])


class AuditLog(Base):
    """Comprehensive audit logging for compliance."""
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"))
    
    # Action details
    action = Column(SQLEnum(AuditAction), nullable=False)
    resource_type = Column(String(50))  # e.g., "user", "budget", "provider"
    resource_id = Column(String(36))
    
    # Context
    description = Column(Text)
    metadata = Column(JSON, default=dict)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Request details (for API calls)
    request_method = Column(String(10))
    request_path = Column(String(500))
    response_status = Column(Integer)
    
    # Timestamp
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="audit_logs")
    user = relationship("User", foreign_keys=[user_id])


class UsageMetrics(Base):
    """Organization usage metrics for billing and analytics."""
    __tablename__ = "usage_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # hour, day, month
    
    # Usage metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Numeric(10, 4), default=0)
    
    # Provider breakdown
    provider_metrics = Column(JSON, default=dict)
    model_metrics = Column(JSON, default=dict)
    
    # Performance metrics
    avg_response_time = Column(Numeric(8, 3))
    p95_response_time = Column(Numeric(8, 3))
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization")


# Indexes for performance
from sqlalchemy import Index

Index('idx_org_users_org_id', OrganizationUser.organization_id)
Index('idx_org_users_user_id', OrganizationUser.user_id)
Index('idx_audit_logs_org_timestamp', AuditLog.organization_id, AuditLog.timestamp)
Index('idx_usage_metrics_org_period', UsageMetrics.organization_id, UsageMetrics.period_start)
Index('idx_organizations_slug', Organization.slug)
Index('idx_users_email', User.email)
