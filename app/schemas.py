from pydantic import BaseModel


class RegisterResponse(BaseModel):
    message: str
    nombre: str
    cedula: str


class RecognizeResponse(BaseModel):
    recognized: bool
    nombre: str | None = None
    cedula: str | None = None
    confidence: float = 0.0
    message: str
