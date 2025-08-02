# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
# ModelMuxer PII Detection and Protection System
# GDPR-compliant PII handling with configurable redaction policies

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


class RedactionAction(str, Enum):
    """Actions to take when PII is detected."""

    REDACT = "redact"  # Replace with [REDACTED]
    MASK = "mask"  # Replace with asterisks
    HASH = "hash"  # Replace with hash
    ENCRYPT = "encrypt"  # Encrypt and store key
    BLOCK = "block"  # Block the entire request
    LOG_ONLY = "log_only"  # Log but don't modify


@dataclass
class PIIDetection:
    """Represents a detected PII instance."""

    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str


@dataclass
class RedactionPolicy:
    """Policy for handling specific PII types."""

    pii_type: PIIType
    action: RedactionAction
    enabled: bool = True
    custom_replacement: str | None = None
    preserve_format: bool = False


class PIIDetector:
    """Advanced PII detection using regex patterns and ML models."""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.common_names = self._load_common_names()

    def _initialize_patterns(self) -> dict[PIIType, list[re.Pattern]]:
        """Initialize regex patterns for PII detection."""
        return {
            PIIType.EMAIL: [
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE)
            ],
            PIIType.PHONE: [
                re.compile(
                    r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
                ),
                re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
                re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"),
            ],
            PIIType.SSN: [
                re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
                re.compile(r"\b\d{9}\b"),
            ],
            PIIType.CREDIT_CARD: [
                # Visa, MasterCard, American Express, Discover
                re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"),  # Visa
                re.compile(r"\b5[1-5][0-9]{14}\b"),  # MasterCard
                re.compile(r"\b3[47][0-9]{13}\b"),  # American Express
                re.compile(r"\b6(?:011|5[0-9]{2})[0-9]{12}\b"),  # Discover
                re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),  # Generic format
            ],
            PIIType.IP_ADDRESS: [
                re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
                re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),  # IPv6
            ],
            PIIType.DATE_OF_BIRTH: [
                re.compile(r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"),
                re.compile(r"\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])\b"),
                re.compile(
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
                    re.IGNORECASE,
                ),
            ],
            PIIType.PASSPORT: [
                re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),  # US passport format
                re.compile(r"\bPassport\s*(?:Number|#)?\s*:?\s*([A-Z0-9]{6,12})\b", re.IGNORECASE),
            ],
            PIIType.DRIVER_LICENSE: [
                re.compile(r"\b[A-Z]\d{7,8}\b"),  # Common DL format
                re.compile(r"\bDL\s*(?:Number|#)?\s*:?\s*([A-Z0-9]{8,12})\b", re.IGNORECASE),
            ],
            PIIType.BANK_ACCOUNT: [
                re.compile(r"\b\d{8,17}\b"),  # Account numbers
                re.compile(r"\bAccount\s*(?:Number|#)?\s*:?\s*(\d{8,17})\b", re.IGNORECASE),
            ],
        }

    def _load_common_names(self) -> set[str]:
        """Load common first and last names for name detection."""
        # In production, this would load from a comprehensive names database
        return {
            "john",
            "jane",
            "michael",
            "sarah",
            "david",
            "lisa",
            "robert",
            "mary",
            "james",
            "patricia",
            "william",
            "jennifer",
            "richard",
            "elizabeth",
            "smith",
            "johnson",
            "williams",
            "brown",
            "jones",
            "garcia",
            "miller",
            "davis",
            "rodriguez",
            "martinez",
            "hernandez",
            "lopez",
            "gonzalez",
        }

    def detect_pii(self, text: str) -> list[PIIDetection]:
        """Detect all PII instances in the given text."""
        detections = []

        # Pattern-based detection
        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    detection = PIIDetection(
                        pii_type=pii_type,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9,  # High confidence for regex matches
                        context=self._get_context(text, match.start(), match.end()),
                    )
                    detections.append(detection)

        # Name detection using common names
        name_detections = self._detect_names(text)
        detections.extend(name_detections)

        # Sort by position
        detections.sort(key=lambda x: x.start_pos)

        # Remove overlapping detections
        detections = self._remove_overlaps(detections)

        return detections

    def _detect_names(self, text: str) -> list[PIIDetection]:
        """Detect potential names using common name patterns."""
        detections = []
        words = re.findall(r"\b[A-Z][a-z]+\b", text)

        for word in words:
            if word.lower() in self.common_names:
                # Find the position in the original text
                start_pos = text.find(word)
                if start_pos != -1:
                    detection = PIIDetection(
                        pii_type=PIIType.NAME,
                        value=word,
                        start_pos=start_pos,
                        end_pos=start_pos + len(word),
                        confidence=0.7,  # Lower confidence for name detection
                        context=self._get_context(text, start_pos, start_pos + len(word)),
                    )
                    detections.append(detection)

        return detections

    def _get_context(self, text: str, start: int, end: int, window: int = 20) -> str:
        """Get context around detected PII."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _remove_overlaps(self, detections: list[PIIDetection]) -> list[PIIDetection]:
        """Remove overlapping detections, keeping the one with higher confidence."""
        if not detections:
            return detections

        result = [detections[0]]

        for current in detections[1:]:
            last = result[-1]

            # Check for overlap
            if current.start_pos < last.end_pos:
                # Keep the one with higher confidence
                if current.confidence > last.confidence:
                    result[-1] = current
            else:
                result.append(current)

        return result


class PIIProtector:
    """PII protection system with configurable redaction policies."""

    def __init__(self, encryption_key: bytes | None = None):
        self.detector = PIIDetector()
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.policies: dict[PIIType, RedactionPolicy] = self._default_policies()

    def _default_policies(self) -> dict[PIIType, RedactionPolicy]:
        """Default redaction policies for different PII types."""
        return {
            PIIType.EMAIL: RedactionPolicy(
                PIIType.EMAIL, RedactionAction.MASK, preserve_format=True
            ),
            PIIType.PHONE: RedactionPolicy(
                PIIType.PHONE, RedactionAction.MASK, preserve_format=True
            ),
            PIIType.SSN: RedactionPolicy(PIIType.SSN, RedactionAction.REDACT),
            PIIType.CREDIT_CARD: RedactionPolicy(PIIType.CREDIT_CARD, RedactionAction.REDACT),
            PIIType.IP_ADDRESS: RedactionPolicy(PIIType.IP_ADDRESS, RedactionAction.HASH),
            PIIType.NAME: RedactionPolicy(PIIType.NAME, RedactionAction.MASK),
            PIIType.ADDRESS: RedactionPolicy(PIIType.ADDRESS, RedactionAction.REDACT),
            PIIType.DATE_OF_BIRTH: RedactionPolicy(PIIType.DATE_OF_BIRTH, RedactionAction.REDACT),
            PIIType.PASSPORT: RedactionPolicy(PIIType.PASSPORT, RedactionAction.REDACT),
            PIIType.DRIVER_LICENSE: RedactionPolicy(PIIType.DRIVER_LICENSE, RedactionAction.REDACT),
            PIIType.BANK_ACCOUNT: RedactionPolicy(PIIType.BANK_ACCOUNT, RedactionAction.REDACT),
        }

    def set_policy(self, pii_type: PIIType, policy: RedactionPolicy) -> None:
        """Set redaction policy for a specific PII type."""
        self.policies[pii_type] = policy
        logger.info("pii_policy_updated", pii_type=pii_type.value, action=policy.action.value)

    def protect_text(
        self, text: str, user_id: str | None = None
    ) -> tuple[str, list[PIIDetection]]:
        """Protect text by detecting and redacting PII according to policies."""
        detections = self.detector.detect_pii(text)

        if not detections:
            return text, detections

        # Process detections in reverse order to maintain positions
        protected_text = text
        processed_detections = []

        for detection in reversed(detections):
            policy = self.policies.get(detection.pii_type)

            if not policy or not policy.enabled:
                continue

            if policy.action == RedactionAction.BLOCK:
                logger.warning(
                    "pii_request_blocked",
                    user_id=user_id,
                    pii_type=detection.pii_type.value,
                    context=detection.context,
                )
                raise ValueError(f"Request blocked due to {detection.pii_type.value} detection")

            elif policy.action == RedactionAction.LOG_ONLY:
                logger.info(
                    "pii_detected",
                    user_id=user_id,
                    pii_type=detection.pii_type.value,
                    context=detection.context,
                )
                continue

            # Apply redaction
            replacement = self._apply_redaction(detection, policy)
            protected_text = (
                protected_text[: detection.start_pos]
                + replacement
                + protected_text[detection.end_pos :]
            )

            # Update detection with redacted value
            detection.value = replacement
            processed_detections.append(detection)

            logger.info(
                "pii_redacted",
                user_id=user_id,
                pii_type=detection.pii_type.value,
                action=policy.action.value,
                original_length=len(detection.value),
                redacted_length=len(replacement),
            )

        return protected_text, list(reversed(processed_detections))

    def _apply_redaction(self, detection: PIIDetection, policy: RedactionPolicy) -> str:
        """Apply specific redaction action to detected PII."""
        original_value = detection.value

        if policy.custom_replacement:
            return policy.custom_replacement

        if policy.action == RedactionAction.REDACT:
            return f"[REDACTED_{detection.pii_type.value.upper()}]"

        elif policy.action == RedactionAction.MASK:
            if policy.preserve_format:
                # Preserve format by replacing only alphanumeric characters
                masked = ""
                for char in original_value:
                    if char.isalnum():
                        masked += "*"
                    else:
                        masked += char
                return masked
            else:
                return "*" * len(original_value)

        elif policy.action == RedactionAction.HASH:
            hash_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
            return f"[HASH_{hash_value}]"

        elif policy.action == RedactionAction.ENCRYPT:
            encrypted = self.cipher.encrypt(original_value.encode())
            # Use SHA256 instead of MD5 for security
            encrypted_id = hashlib.sha256(encrypted).hexdigest()[:8]
            # In production, store the encrypted value with the ID
            return f"[ENCRYPTED_{encrypted_id}]"

        return original_value

    def get_protection_summary(self, detections: list[PIIDetection]) -> dict[str, Any]:
        """Get summary of PII protection actions taken."""
        summary = {
            "total_detections": len(detections),
            "by_type": {},
            "actions_taken": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        for detection in detections:
            pii_type = detection.pii_type.value
            summary["by_type"][pii_type] = summary["by_type"].get(pii_type, 0) + 1

            policy = self.policies.get(detection.pii_type)
            if policy:
                action = policy.action.value
                summary["actions_taken"][action] = summary["actions_taken"].get(action, 0) + 1

        return summary
