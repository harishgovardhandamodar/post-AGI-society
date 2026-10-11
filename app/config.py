"""Environment-based configuration (12-factor).

All values have safe development defaults so the app boots with zero setup,
but every knob can be overridden via environment variables — no code edits
required for deployment.
"""
import os


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Database -------------------------------------------------------------
DATA_DIR = os.environ.get("AGS_DATA_DIR", "data")
DATABASE_URL = os.environ.get(
    "AGS_DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'ags.db')}"
)

# --- Security / auth ------------------------------------------------------
# Shared secret for mutating endpoints. Empty string = open (dev mode);
# set AGS_ADMIN_TOKEN in production and send it as `X-Admin-Token` or
# `Authorization: Bearer <token>`.
ADMIN_TOKEN = os.environ.get("AGS_ADMIN_TOKEN", "")

# CORS: comma-separated list of allowed origins. "*" keeps the permissive dev
# default but credentials are then disabled automatically (see main.py), since
# wildcard origin + allow_credentials is both invalid and unsafe.
CORS_ORIGINS = [o.strip() for o in os.environ.get("AGS_CORS_ORIGINS", "*").split(",") if o.strip()]

# --- Ingestion scheduler ---------------------------------------------------
INGEST_ENABLED = _bool("AGS_INGEST_ENABLED", True)
INGEST_HOUR = int(os.environ.get("AGS_INGEST_HOUR", "4"))
INGEST_MINUTE = int(os.environ.get("AGS_INGEST_MINUTE", "0"))

# --- Seeding ----------------------------------------------------------------
SEED_ON_STARTUP = _bool("AGS_SEED_ON_STARTUP", True)
