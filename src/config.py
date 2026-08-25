from pathlib import Path

from dotenv import load_dotenv


def load_env(env: str = "local"):
    base = Path(__file__).resolve().parent.parent
    path = base / f".env.{env}"
    load_dotenv(dotenv_path=path)
