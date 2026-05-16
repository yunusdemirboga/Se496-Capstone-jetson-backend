from app.config import settings

def generate_signed_url(file_path: str) -> str | None:
    if not file_path:
        return None
    return f"{settings.SUPABASE_URL}/storage/v1/object/public/{settings.SUPABASE_BUCKET}/{file_path}"