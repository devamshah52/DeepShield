from datetime import timezone

from fastapi import APIRouter, Depends

from app.utils.jwt_handler import get_current_user
from app.database.mongodb import uploads_collection

router = APIRouter(
    prefix="/history",
    tags=["History"],
)


@router.get("/")
async def get_history(
    current_user: dict = Depends(get_current_user),
):

    uploads = uploads_collection.find(
        {
            "user_id": str(current_user["_id"])
        }
    ).sort(
        "uploaded_at",
        -1
    )

    history = []

    for upload in uploads:

        uploaded_at = upload.get("uploaded_at")

        history.append(

            {

                "id": str(upload["_id"]),

                "original_filename": upload["original_filename"],

                "uploaded_at": (
                    uploaded_at.replace(tzinfo=timezone.utc).isoformat()
                    if uploaded_at
                    else None
                ),

                "status": upload["status"],

                "prediction": upload.get("prediction"),

                "confidence": upload.get("confidence"),

            }

        )

    return history