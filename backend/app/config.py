from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_BUCKET: str
    SIGNED_URL_EXPIRY_SECONDS: int = 3600

    class Config:
        env_file = ".env"

settings = Settings()