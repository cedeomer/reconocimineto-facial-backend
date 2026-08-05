import io

import face_recognition
import numpy as np
from PIL import Image

MATCH_TOLERANCE = 0.45


def load_image_from_bytes(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(image)


def extract_face_encoding(image_bytes: bytes) -> np.ndarray:
    rgb = load_image_from_bytes(image_bytes)
    locations = face_recognition.face_locations(rgb)
    if not locations:
        raise ValueError("No se detectó ningún rostro en la imagen.")
    if len(locations) > 1:
        raise ValueError("Se detectaron múltiples rostros. Use una sola persona.")
    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
    return encodings[0]


def encoding_to_bytes(encoding: np.ndarray) -> bytes:
    return encoding.astype(np.float64).tobytes()


def bytes_to_encoding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float64)


def recognize_face(
    image_bytes: bytes,
    known_encodings: list[np.ndarray],
    tolerance: float = MATCH_TOLERANCE,
) -> tuple[int | None, float]:
    rgb = load_image_from_bytes(image_bytes)
    unknown_locations = face_recognition.face_locations(rgb)
    if not unknown_locations:
        return None, 0.0

    unknown_encodings = face_recognition.face_encodings(rgb, unknown_locations)
    if not unknown_encodings:
        return None, 0.0

    unknown = unknown_encodings[0]
    if not known_encodings:
        return None, 0.0

    distances = face_recognition.face_distance(known_encodings, unknown)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    confidence = max(0.0, 1.0 - best_distance)

    if best_distance <= tolerance:
        return best_idx, confidence
    return None, confidence
