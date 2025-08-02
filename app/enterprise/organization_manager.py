# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
# ModelMuxer Organization Management Service
# Comprehensive multi-tenant organization management

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import secrets
import hashlib

from .models import (
    Organization, User, OrganizationUser, OrganizationBudget,
    OrganizationProvider, OrganizationAPIKey, AuditLog, UsageMetrics,
    PlanType, OrganizationStatus, UserRole, AuditAction
)
from ..security.auth import SecurityManager

logger = structlog.get_logger(__name__)


class OrganizationManager:
    """Comprehensive organization management for multi-tenancy."""
    
    def __init__(self, db_session: Session, security_manager: SecurityManager):
        self.db = db_session
        self.security_manager = security_manager
        
        # Plan limits configuration
        self.plan_limits = {
            PlanType.FREE: {
                "monthly_requests": 1000,
                "monthly_cost": 10.0,
                "max_users": 3,
                "max_api_keys": 2,
                "max_providers": 2,
                "features": ["basic_routing", "cost_tracking"]
            },
            PlanType.STARTER: {
                "monthly_requests": 10000,
                "monthly_cost": 100.0,
                "max_users": 10,
                "max_api_keys": 10,
                "max_providers": 5,
                "features": ["basic_routing", "cost_tracking", "analytics", "cascade_routing"]
            },
            PlanType.PROFESSIONAL: {
                "monthly_requests": 100000,
                "monthly_cost": 1000.0,
                "max_users": 50,
                "max_api_keys": 50,
                "max_providers": 10,
                "features": ["all_routing", "advanced_analytics", "custom_models", "priority_support"]
            },
            PlanType.ENTERPRISE: {
                "monthly_requests": 1000000,
                "monthly_cost": 10000.0,
                "max_users": 500,
                "max_api_keys": 200,
                "max_providers": 50,
                "features": ["all_features", "custom_deployment", "dedicated_support", "sla"]
            }
        }
    
    async def create_organization(
        self,
        name: str,
        slug: str,
        owner_email: str,
        plan_type: PlanType = PlanType.FREE,
        trial_days: int = 14
    ) -> Tuple[Organization, User]:
        """Create a new organization with owner."""
        try:
            # Check if slug is available
            existing_org = self.db.query(Organization).filter(Organization.slug == slug).first()
            if existing_org:
                raise ValueError(f"Organization slug '{slug}' is already taken")
            
            # Get or create owner user
            owner = self.db.query(User).filter(User.email == owner_email).first()
            if not owner:
                # Create new user (password will be set via invitation)
                temp_password = secrets.token_urlsafe(32)
                owner = User(
                    email=owner_email,
                    password_hash=self.security_manager.hash_password(temp_password),
                    is_verified=False
                )
                self.db.add(owner)
                self.db.flush()  # Get user ID
            
            # Create organization
            plan_config = self.plan_limits[plan_type]
            trial_end = datetime.utcnow() + timedelta(days=trial_days) if trial_days > 0 else None
            
            organization = Organization(
                name=name,
                slug=slug,
                plan_type=plan_type,
                status=OrganizationStatus.TRIAL if trial_days > 0 else OrganizationStatus.ACTIVE,
                trial_ends_at=trial_end,
                monthly_request_limit=plan_config["monthly_requests"],
                monthly_cost_limit=plan_config["monthly_cost"],
                max_users=plan_config["max_users"],
                max_api_keys=plan_config["max_api_keys"],
                created_by=owner.id,
                settings={
                    "features": plan_config["features"],
                    "max_providers": plan_config["max_providers"]
                }
            )
            self.db.add(organization)
            self.db.flush()  # Get organization ID
            
            # Add owner to organization
            org_user = OrganizationUser(
                organization_id=organization.id,
                user_id=owner.id,
                role=UserRole.OWNER,
                is_active=True,
                joined_at=datetime.utcnow()
            )
            self.db.add(org_user)
            
            # Create default budget
            default_budget = OrganizationBudget(
                organization_id=organization.id,
                budget_type="monthly",
                budget_limit=plan_config["monthly_cost"],
                alert_thresholds=[50.0, 80.0, 95.0],
                created_by=owner.id
            )
            self.db.add(default_budget)
            
            # Log audit event
            await self._log_audit_event(
                organization.id,
                owner.id,
                AuditAction.CREATE,
                "organization",
                organization.id,
                f"Organization '{name}' created with {plan_type.value} plan"
            )
            
            self.db.commit()
            
            logger.info("organization_created",
                       org_id=organization.id,
                       name=name,
                       slug=slug,
                       plan=plan_type.value,
                       owner_email=owner_email)
            
            return organization, owner
            
        except Exception as e:
            self.db.rollback()
            logger.error("organization_creation_failed", error=str(e), name=name, slug=slug)
            raise
    
    async def invite_user(
        self,
        organization_id: str,
        inviter_id: str,
        email: str,
        role: UserRole,
        permissions: Optional[List[str]] = None
    ) -> OrganizationUser:
        """Invite a user to an organization."""
        try:
            # Check organization limits
            org = self.db.query(Organization).filter(Organization.id == organization_id).first()
            if not org:
                raise ValueError("Organization not found")
            
            current_users = self.db.query(OrganizationUser).filter(
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.is_active == True
            ).count()
            
            if current_users >= org.max_users:
                raise ValueError(f"Organization has reached maximum user limit ({org.max_users})")
            
            # Check if user already exists in organization
            existing_membership = self.db.query(OrganizationUser).filter(
                OrganizationUser.organization_id == organization_id,
                OrganizationUser.user_id.in_(
                    self.db.query(User.id).filter(User.email == email)
                )
            ).first()
            
            if existing_membership:
                raise ValueError("User is already a member of this organization")
            
            # Get or create user
            user = self.db.query(User).filter(User.email == email).first()
            if not user:
                temp_password = secrets.token_urlsafe(32)
                user = User(
                    email=email,
                    password_hash=self.security_manager.hash_password(temp_password),
                    is_verified=False
                )
                self.db.add(user)
                self.db.flush()
            
            # Create organization membership
            org_user = OrganizationUser(
                organization_id=organization_id,
                user_id=user.id,
                role=role,
                permissions=permissions or [],
                is_active=True,
                invited_by=inviter_id,
                invited_at=datetime.utcnow()
            )
            self.db.add(org_user)
            
            # Log audit event
            await self._log_audit_event(
                organization_id,
                inviter_id,
                AuditAction.USER_INVITED,
                "user",
                user.id,
                f"User {email} invited with role {role.value}"
            )
            
            self.db.commit()
            
            logger.info("user_invited",
                       org_id=organization_id,
                       user_email=email,
                       role=role.value,
                       inviter_id=inviter_id)
            
            return org_user
            
        except Exception as e:
            self.db.rollback()
            logger.error("user_invitation_failed", error=str(e), org_id=organization_id, email=email)
            raise
    
    async def create_api_key(
        self,
        organization_id: str,
        creator_id: str,
        name: str,
        scopes: List[str],
        description: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limits: Optional[Dict[str, int]] = None
    ) -> Tuple[str, OrganizationAPIKey]:
        """Create an API key for an organization."""
        try:
            # Check organization limits
            org = self.db.query(Organization).filter(Organization.id == organization_id).first()
            if not org:
                raise ValueError("Organization not found")
            
            current_keys = self.db.query(OrganizationAPIKey).filter(
                OrganizationAPIKey.organization_id == organization_id,
                OrganizationAPIKey.is_active == True
            ).count()
            
            if current_keys >= org.max_api_keys:
                raise ValueError(f"Organization has reached maximum API key limit ({org.max_api_keys})")
            
            # Generate API key
            api_key = f"mm_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Set expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Create API key record
            api_key_record = OrganizationAPIKey(
                organization_id=organization_id,
                key_hash=key_hash,
                name=name,
                description=description,
                scopes=scopes,
                expires_at=expires_at,
                rate_limit_per_minute=rate_limits.get("per_minute", 60) if rate_limits else 60,
                rate_limit_per_hour=rate_limits.get("per_hour", 1000) if rate_limits else 1000,
                monthly_cost_limit=rate_limits.get("monthly_cost") if rate_limits else None,
                created_by=creator_id
            )
            self.db.add(api_key_record)
            
            # Log audit event
            await self._log_audit_event(
                organization_id,
                creator_id,
                AuditAction.CREATE,
                "api_key",
                api_key_record.id,
                f"API key '{name}' created with scopes: {', '.join(scopes)}"
            )
            
            self.db.commit()
            
            logger.info("api_key_created",
                       org_id=organization_id,
                       key_name=name,
                       scopes=scopes,
                       creator_id=creator_id)
            
            return api_key, api_key_record
            
        except Exception as e:
            self.db.rollback()
            logger.error("api_key_creation_failed", error=str(e), org_id=organization_id, name=name)
            raise
    
    async def get_organization_usage(
        self,
        organization_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Get comprehensive usage statistics for an organization."""
        try:
            # Get usage metrics
            usage_metrics = self.db.query(UsageMetrics).filter(
                UsageMetrics.organization_id == organization_id,
                UsageMetrics.period_start >= period_start,
                UsageMetrics.period_end <= period_end
            ).all()
            
            # Aggregate metrics
            total_requests = sum(m.total_requests for m in usage_metrics)
            total_cost = sum(m.total_cost for m in usage_metrics)
            total_tokens = sum(m.total_tokens for m in usage_metrics)
            
            # Provider breakdown
            provider_breakdown = {}
            model_breakdown = {}
            
            for metric in usage_metrics:
                if metric.provider_metrics:
                    for provider, stats in metric.provider_metrics.items():
                        if provider not in provider_breakdown:
                            provider_breakdown[provider] = {"requests": 0, "cost": 0, "tokens": 0}
                        provider_breakdown[provider]["requests"] += stats.get("requests", 0)
                        provider_breakdown[provider]["cost"] += stats.get("cost", 0)
                        provider_breakdown[provider]["tokens"] += stats.get("tokens", 0)
                
                if metric.model_metrics:
                    for model, stats in metric.model_metrics.items():
                        if model not in model_breakdown:
                            model_breakdown[model] = {"requests": 0, "cost": 0, "tokens": 0}
                        model_breakdown[model]["requests"] += stats.get("requests", 0)
                        model_breakdown[model]["cost"] += stats.get("cost", 0)
                        model_breakdown[model]["tokens"] += stats.get("tokens", 0)
            
            # Get organization limits
            org = self.db.query(Organization).filter(Organization.id == organization_id).first()
            
            return {
                "period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat()
                },
                "usage": {
                    "total_requests": total_requests,
                    "total_cost": float(total_cost),
                    "total_tokens": total_tokens,
                    "avg_cost_per_request": float(total_cost / total_requests) if total_requests > 0 else 0
                },
                "limits": {
                    "monthly_requests": org.monthly_request_limit,
                    "monthly_cost": float(org.monthly_cost_limit),
                    "request_utilization": (total_requests / org.monthly_request_limit) * 100 if org.monthly_request_limit > 0 else 0,
                    "cost_utilization": (float(total_cost) / float(org.monthly_cost_limit)) * 100 if org.monthly_cost_limit > 0 else 0
                },
                "breakdown": {
                    "by_provider": provider_breakdown,
                    "by_model": model_breakdown
                }
            }
            
        except Exception as e:
            logger.error("usage_retrieval_failed", error=str(e), org_id=organization_id)
            raise
    
    async def _log_audit_event(
        self,
        organization_id: str,
        user_id: Optional[str],
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log an audit event."""
        try:
            audit_log = AuditLog(
                organization_id=organization_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                description=description,
                metadata=metadata or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            self.db.add(audit_log)
            # Note: Commit is handled by the calling method
            
        except Exception as e:
            logger.error("audit_logging_failed", error=str(e), org_id=organization_id)
    
    async def check_organization_limits(
        self,
        organization_id: str,
        check_type: str,
        current_usage: Optional[int] = None
    ) -> Dict[str, Any]:
        """Check if organization is within limits."""
        try:
            org = self.db.query(Organization).filter(Organization.id == organization_id).first()
            if not org:
                raise ValueError("Organization not found")
            
            limits_status = {
                "within_limits": True,
                "warnings": [],
                "errors": []
            }
            
            if check_type == "users":
                current_users = self.db.query(OrganizationUser).filter(
                    OrganizationUser.organization_id == organization_id,
                    OrganizationUser.is_active == True
                ).count()
                
                if current_users >= org.max_users:
                    limits_status["within_limits"] = False
                    limits_status["errors"].append(f"User limit exceeded: {current_users}/{org.max_users}")
                elif current_users >= org.max_users * 0.8:
                    limits_status["warnings"].append(f"Approaching user limit: {current_users}/{org.max_users}")
            
            elif check_type == "api_keys":
                current_keys = self.db.query(OrganizationAPIKey).filter(
                    OrganizationAPIKey.organization_id == organization_id,
                    OrganizationAPIKey.is_active == True
                ).count()
                
                if current_keys >= org.max_api_keys:
                    limits_status["within_limits"] = False
                    limits_status["errors"].append(f"API key limit exceeded: {current_keys}/{org.max_api_keys}")
                elif current_keys >= org.max_api_keys * 0.8:
                    limits_status["warnings"].append(f"Approaching API key limit: {current_keys}/{org.max_api_keys}")
            
            elif check_type == "requests" and current_usage is not None:
                if current_usage >= org.monthly_request_limit:
                    limits_status["within_limits"] = False
                    limits_status["errors"].append(f"Request limit exceeded: {current_usage}/{org.monthly_request_limit}")
                elif current_usage >= org.monthly_request_limit * 0.8:
                    limits_status["warnings"].append(f"Approaching request limit: {current_usage}/{org.monthly_request_limit}")
            
            return limits_status
            
        except Exception as e:
            logger.error("limits_check_failed", error=str(e), org_id=organization_id)
            raise
