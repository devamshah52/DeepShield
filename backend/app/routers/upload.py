from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
)

from datetime import datetime, timezone

from pathlib import Path

import shutil
import uuid

from app.utils.jwt_handler import (
    get_current_user,
)

from app.database.mongodb import (
    uploads_collection,
)

from app.services.predict import (
    predict_audio,
)


router = APIRouter()


# ============================================================
# Supported formats
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


# ============================================================
# Upload Audio
# ============================================================

@router.post("/upload")
async def upload_audio(

    audio: UploadFile = File(...),

    current_user: dict = Depends(
        get_current_user
    ),

):

    # ========================================================
    # Validate filename
    # ========================================================

    if not audio.filename:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail="No audio filename was provided.",

        )

    extension = Path(
        audio.filename
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=(
                "Unsupported audio format. "
                "Supported formats: "
                "WAV, MP3, M4A, FLAC, OGG."
            ),

        )

    # ========================================================
    # User Upload Folder
    # ========================================================

    user_folder = (
        Path("uploads")
        / str(current_user["_id"])
    )

    user_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Generate unique filename
    # ========================================================

    unique_filename = (
        f"{uuid.uuid4()}{extension}"
    )

    file_path = (
        user_folder
        / unique_filename
    )

    # ========================================================
    # Save uploaded file
    # ========================================================

    try:

        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                audio.file,
                buffer,
            )

    except Exception as error:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=(
                f"Failed to save audio file: "
                f"{str(error)}"
            ),

        )

    # ========================================================
    # Create MongoDB upload record
    # ========================================================

    upload_document = {

        "user_id": str(
            current_user["_id"]
        ),

        "filename": unique_filename,

        "original_filename": (
            audio.filename
        ),

        "filepath": str(
            file_path
        ),

        "uploaded_at": (
            datetime.now(timezone.utc)
        ),

        "prediction": None,

        "confidence": None,

        "probabilities": None,

        "feature_shape": None,

        "device": None,

        "status": "uploaded",

    }

    result = uploads_collection.insert_one(
        upload_document
    )

    upload_id = result.inserted_id

    # ========================================================
    # Run DeepShield AI analysis
    # ========================================================

    try:

        prediction_result = predict_audio(
            str(file_path)
        )

        # ----------------------------------------------------
        # Update MongoDB
        # ----------------------------------------------------

        uploads_collection.update_one(

            {
                "_id": upload_id
            },

            {
                "$set": {

                    "prediction": (
                        prediction_result[
                            "prediction"
                        ]
                    ),

                    "confidence": (
                        prediction_result[
                            "confidence"
                        ]
                    ),

                    "probabilities": (
                        prediction_result[
                            "probabilities"
                        ]
                    ),

                    "feature_shape": (
                        prediction_result[
                            "feature_shape"
                        ]
                    ),

                    "device": (
                        prediction_result[
                            "device"
                        ]
                    ),

                    "status": "completed",

                }
            },

        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "message": (
                "Audio analyzed successfully."
            ),

            "upload_id": str(
                upload_id
            ),

            "filename": unique_filename,

            "original_filename": (
                audio.filename
            ),

            "status": "completed",

            "prediction": (
                prediction_result[
                    "prediction"
                ]
            ),

            "confidence": (
                prediction_result[
                    "confidence"
                ]
            ),

            "probabilities": (
                prediction_result[
                    "probabilities"
                ]
            ),

        }

    except Exception as error:

        # ----------------------------------------------------
        # Update failed status
        # ----------------------------------------------------

        uploads_collection.update_one(

            {
                "_id": upload_id
            },

            {
                "$set": {

                    "status": "failed",

                    "error": str(error),

                }
            },

        )

        raise HTTPException(

            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),

            detail=(
                "Audio was uploaded, "
                "but AI analysis failed."
            ),

        )