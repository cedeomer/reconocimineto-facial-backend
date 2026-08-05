import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

load_dotenv()

from .database import Base, engine, get_db
from .face_service import (
    bytes_to_encoding,
    encoding_to_bytes,
    extract_face_encoding,
    recognize_face,
)
from .models import User
from .schemas import RecognizeResponse, RegisterResponse

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Facial Recognition API", version="1.0.0")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/register", response_model=RegisterResponse)
async def register_user(
    nombre: str = Form(...),
    cedula: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    existing = db.query(User).filter(User.cedula == cedula.strip()).first()
    if existing:
        raise HTTPException(status_code=409, detail="La cédula ya está registrada.")

    image_bytes = await image.read()
    try:
        encoding = extract_face_encoding(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user = User(
        nombre=nombre.strip(),
        cedula=cedula.strip(),
        encoding=encoding_to_bytes(encoding),
    )
    db.add(user)
    db.commit()

    return RegisterResponse(
        message="Usuario registrado correctamente.",
        nombre=user.nombre,
        cedula=user.cedula,
    )


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_user(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    users = db.query(User).all()
    if not users:
        return RecognizeResponse(
            recognized=False,
            message="Unknown/No reconocido",
        )

    known_encodings = [bytes_to_encoding(user.encoding) for user in users]
    image_bytes = await image.read()

    try:
        match_idx, confidence = recognize_face(image_bytes, known_encodings)
    except Exception:
        return RecognizeResponse(
            recognized=False,
            message="Unknown/No reconocido",
            confidence=0.0,
        )

    if match_idx is None:
        return RecognizeResponse(
            recognized=False,
            message="Unknown/No reconocido",
            confidence=confidence,
        )

    matched = users[match_idx]
    return RecognizeResponse(
        recognized=True,
        nombre=matched.nombre,
        cedula=matched.cedula,
        confidence=confidence,
        message="Usuario reconocido",
    )
