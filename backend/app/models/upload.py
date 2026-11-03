from pydantic import BaseModel


class UploadResponse(BaseModel):

    id: str

    filename: str

    original_filename: str

    uploaded_at: str

    status: str