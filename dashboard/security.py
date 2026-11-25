"""Part 14.2 security for the platform console: session auth, roles
(viewer < trader < admin), TOTP MFA, and rate limiting — all stdlib,
fail-closed. Single-user tool (explicit decision recorded in the
operations doc); no multi-tenancy, no credential storage beyond the
operator password hash and TOTP secret, both from the environment only.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
import threading
import time

# ── Roles ────────────────────────────────────────────────────────────────────

ROLE_LEVELS = {"viewer": 1, "trader": 2, "admin": 3}


def role_at_least(role: str, required: str) -> bool:
    return ROLE_LEVELS.get(role, 0) >= ROLE_LEVELS[required]


# ── Password hashing (PBKDF2-HMAC-SHA256, stdlib) ────────────────────────────

def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), 200_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


# ── TOTP (RFC 6238, stdlib only) ─────────────────────────────────────────────

def generate_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode()


def totp_code(secret_b32: str, at_time: int | None = None,
              step: int = 30, digits: int = 6) -> str:
    counter = int((at_time if at_time is not None else time.time()) // step)
    key = base64.b32decode(secret_b32, casefold=True)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = (struct.unpack(">I", digest[offset:offset + 4])[0]
             & 0x7FFFFFFF) % (10 ** digits)
    return str(value).zfill(digits)


def verify_totp(secret_b32: str, code: str,
                at_time: int | None = None, window: int = 1) -> bool:
    """Accepts the current step ± window; constant-time comparison."""
    if not code or not code.isdigit():
        return False
    t = at_time if at_time is not None else time.time()
    for drift in range(-window, window + 1):
        if hmac.compare_digest(totp_code(secret_b32, t + drift * 30), code):
            return True
    return False


# ── Sessions ─────────────────────────────────────────────────────────────────

class SessionStore:
    """In-memory sessions with expiry and revocation. Tokens are random
    (not derivable); a restart logs everyone out — acceptable and honest
    for a single-user tool."""

    def __init__(self, ttl_s: int = 8 * 3600):
        self.ttl_s = ttl_s
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, username: str, role: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {"username": username, "role": role,
                                     "created": time.time(),
                                     "expires": time.time() + self.ttl_s}
        return token

    def get(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self._lock:
            sess = self._sessions.get(token)
            if sess is None or sess["expires"] < time.time():
                self._sessions.pop(token, None)
                return None
            return dict(sess)

    def revoke(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)


# ── Rate limiting (sliding window, per key) ──────────────────────────────────

class RateLimiter:
    def __init__(self, max_events: int = 5, window_s: int = 60):
        self.max_events = max_events
        self.window_s = window_s
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = [t for t in self._events.get(key, [])
                      if now - t < self.window_s]
            if len(bucket) >= self.max_events:
                self._events[key] = bucket
                return False
            bucket.append(now)
            self._events[key] = bucket
            return True


# ── Authenticator wiring ─────────────────────────────────────────────────────

class Authenticator:
    """Environment-configured operator account:
    DSH_ADMIN_USER (default 'admin'), DSH_ADMIN_PASSWORD (required —
    without it every MUTATING endpoint fails closed with
    auth_not_configured), DSH_TOTP_SECRET (optional but recommended)."""

    def __init__(self):
        self.username = os.environ.get("DSH_ADMIN_USER", "admin")
        self._password_hash = None
        self._password_raw = os.environ.get("DSH_ADMIN_PASSWORD")
        self.totp_secret = os.environ.get("DSH_TOTP_SECRET")
        self.sessions = SessionStore()
        self.login_limiter = RateLimiter(max_events=5, window_s=60)
        if self._password_raw:
            self._password_hash = hash_password(self._password_raw)
            self._password_raw = None          # never retained

    @property
    def configured(self) -> bool:
        return self._password_hash is not None

    def login(self, username: str, password: str, totp: str | None,
              client_key: str = "ip") -> tuple[str | None, str | None]:
        """Returns (token, error). Rate-limited fail closed."""
        if not self.login_limiter.allow(client_key):
            return None, "rate_limited"
        if not self.configured:
            return None, "auth_not_configured"
        if username != self.username or \
                not verify_password(password, self._password_hash):
            return None, "invalid_credentials"
        if self.totp_secret and not verify_totp(self.totp_secret, totp or ""):
            return None, "invalid_totp"
        return self.sessions.create(username, "admin"), None


AUTH = Authenticator()
