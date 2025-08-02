# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
# ModelMuxer Authentication and Authorization System
# Production-grade JWT-based authentication with RBAC

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from enum import Enum
import secrets
import structlog
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
from sqlalchemy.orm import Session

logger = structlog.get_logger(__name__)

# Security configuration
JWT_ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
PASSWORD_MIN_LENGTH = 12
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    ORG_ADMIN = "org_admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Granular permissions for RBAC."""
    # System permissions
    FULL_ACCESS = "full_access"
    SYSTEM_CONFIG = "system_config"
    USER_MANAGEMENT = "user_management"
    
    # Organization permissions
    ORG_USERS = "org_users"
    ORG_BUDGETS = "org_budgets"
    ORG_ANALYTICS = "org_analytics"
    ORG_PROVIDERS = "org_providers"
    
    # API permissions
    API_ACCESS = "api_access"
    API_KEY_MANAGEMENT = "manage_api_keys"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view_analytics"
    VIEW_OWN_USAGE = "view_own_usage"
    VIEW_OWN_BUDGETS = "view_own_budgets"
    
    # Provider permissions
    PROVIDER_CONFIG = "provider_config"
    MODEL_ACCESS = "model_access"


# Role-based permission mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.FULL_ACCESS,
        Permission.SYSTEM_CONFIG,
        Permission.USER_MANAGEMENT,
        Permission.ORG_USERS,
        Permission.ORG_BUDGETS,
        Permission.ORG_ANALYTICS,
        Permission.ORG_PROVIDERS,
        Permission.API_ACCESS,
        Permission.API_KEY_MANAGEMENT,
        Permission.VIEW_ANALYTICS,
        Permission.PROVIDER_CONFIG,
        Permission.MODEL_ACCESS,
    },
    UserRole.ORG_ADMIN: {
        Permission.ORG_USERS,
        Permission.ORG_BUDGETS,
        Permission.ORG_ANALYTICS,
        Permission.ORG_PROVIDERS,
        Permission.API_ACCESS,
        Permission.API_KEY_MANAGEMENT,
        Permission.VIEW_ANALYTICS,
        Permission.MODEL_ACCESS,
    },
    UserRole.DEVELOPER: {
        Permission.API_ACCESS,
        Permission.API_KEY_MANAGEMENT,
        Permission.VIEW_ANALYTICS,
        Permission.VIEW_OWN_USAGE,
        Permission.VIEW_OWN_BUDGETS,
        Permission.MODEL_ACCESS,
    },
    UserRole.VIEWER: {
        Permission.VIEW_OWN_USAGE,
        Permission.VIEW_OWN_BUDGETS,
    },
}


class AuthenticationError(Exception):
    """Custom authentication error."""
    pass


class AuthorizationError(Exception):
    """Custom authorization error."""
    pass


class SecurityManager:
    """Comprehensive security manager for authentication and authorization."""
    
    def __init__(self, redis_client: redis.Redis, private_key: str, public_key: str):
        self.redis_client = redis_client
        self.private_key = private_key
        self.public_key = public_key
        self.security_bearer = HTTPBearer()
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        if len(password) < PASSWORD_MIN_LENGTH:
            raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long")
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def generate_api_key(self, user_id: str, scopes: List[str]) -> str:
        """Generate API key with scopes."""
        api_key = f"mm_{secrets.token_urlsafe(32)}"
        
        # Store API key metadata in Redis
        key_data = {
            "user_id": user_id,
            "scopes": ",".join(scopes),
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "usage_count": 0
        }
        
        self.redis_client.hset(f"api_key:{api_key}", mapping=key_data)
        self.redis_client.expire(f"api_key:{api_key}", 86400 * 365)  # 1 year
        
        logger.info("api_key_generated", user_id=user_id, scopes=scopes)
        return api_key
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key and return user info."""
        key_data = self.redis_client.hgetall(f"api_key:{api_key}")
        
        if not key_data:
            raise AuthenticationError("Invalid API key")
        
        # Update usage statistics
        self.redis_client.hincrby(f"api_key:{api_key}", "usage_count", 1)
        self.redis_client.hset(f"api_key:{api_key}", "last_used", datetime.utcnow().isoformat())
        
        return {
            "user_id": key_data["user_id"],
            "scopes": key_data["scopes"].split(",") if key_data["scopes"] else [],
            "auth_method": "api_key"
        }
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        payload = {
            "sub": user_data["user_id"],
            "email": user_data.get("email"),
            "role": user_data.get("role", UserRole.VIEWER),
            "org_id": user_data.get("org_id"),
            "permissions": list(ROLE_PERMISSIONS.get(UserRole(user_data.get("role", UserRole.VIEWER)), set())),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.private_key, algorithm=JWT_ALGORITHM)
        
        # Store token in Redis for revocation capability
        self.redis_client.setex(
            f"access_token:{user_data['user_id']}:{token[-10:]}",
            ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "valid"
        )
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.private_key, algorithm=JWT_ALGORITHM)
        
        # Store refresh token
        self.redis_client.setex(
            f"refresh_token:{user_id}:{payload['jti']}",
            REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
            token
        )
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.public_key, algorithms=[JWT_ALGORITHM])
            
            # Check if token is revoked
            if payload.get("type") == "access":
                token_key = f"access_token:{payload['sub']}:{token[-10:]}"
                if not self.redis_client.exists(token_key):
                    raise AuthenticationError("Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def revoke_token(self, token: str) -> None:
        """Revoke a specific token."""
        try:
            payload = jwt.decode(token, self.public_key, algorithms=[JWT_ALGORITHM])
            
            if payload.get("type") == "access":
                token_key = f"access_token:{payload['sub']}:{token[-10:]}"
                self.redis_client.delete(token_key)
            elif payload.get("type") == "refresh":
                token_key = f"refresh_token:{payload['sub']}:{payload['jti']}"
                self.redis_client.delete(token_key)
                
            logger.info("token_revoked", user_id=payload.get("sub"), token_type=payload.get("type"))
            
        except jwt.InvalidTokenError:
            pass  # Token is already invalid
    
    def revoke_all_user_tokens(self, user_id: str) -> None:
        """Revoke all tokens for a user."""
        # Delete all access tokens
        access_pattern = f"access_token:{user_id}:*"
        for key in self.redis_client.scan_iter(match=access_pattern):
            self.redis_client.delete(key)
        
        # Delete all refresh tokens
        refresh_pattern = f"refresh_token:{user_id}:*"
        for key in self.redis_client.scan_iter(match=refresh_pattern):
            self.redis_client.delete(key)
        
        logger.info("all_tokens_revoked", user_id=user_id)
    
    def check_rate_limit(self, user_id: str, action: str, limit: int, window: int) -> bool:
        """Check rate limiting for user actions."""
        key = f"rate_limit:{user_id}:{action}"
        current = self.redis_client.incr(key)
        
        if current == 1:
            self.redis_client.expire(key, window)
        
        return current <= limit
    
    def record_login_attempt(self, user_id: str, success: bool, ip_address: str) -> None:
        """Record login attempt for security monitoring."""
        attempt_key = f"login_attempts:{user_id}"
        
        if success:
            # Clear failed attempts on successful login
            self.redis_client.delete(attempt_key)
            logger.info("login_success", user_id=user_id, ip_address=ip_address)
        else:
            # Increment failed attempts
            attempts = self.redis_client.incr(attempt_key)
            self.redis_client.expire(attempt_key, LOCKOUT_DURATION_MINUTES * 60)
            
            if attempts >= MAX_LOGIN_ATTEMPTS:
                # Lock account
                self.redis_client.setex(f"account_locked:{user_id}", LOCKOUT_DURATION_MINUTES * 60, "locked")
                logger.warning("account_locked", user_id=user_id, attempts=attempts, ip_address=ip_address)
            else:
                logger.warning("login_failed", user_id=user_id, attempts=attempts, ip_address=ip_address)
    
    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed login attempts."""
        return self.redis_client.exists(f"account_locked:{user_id}")
    
    def has_permission(self, user_role: UserRole, required_permission: Permission) -> bool:
        """Check if user role has required permission."""
        user_permissions = ROLE_PERMISSIONS.get(user_role, set())
        return required_permission in user_permissions or Permission.FULL_ACCESS in user_permissions
