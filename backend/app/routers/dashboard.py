from fastapi import APIRouter, Depends

from datetime import timezone

from app.models.dashboard import (
    DashboardResponse,
    DashboardStatistics,
    RecentActivity,
)

from app.utils.jwt_handler import get_current_user

from app.database.mongodb import uploads_collection


router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)


# ============================================================
# GET DASHBOARD
# ============================================================

@router.get(
    "/",
    response_model=DashboardResponse,
)
async def get_dashboard(
    current_user: dict = Depends(get_current_user),
):
    """
    Return dashboard statistics and recent activity
    for the currently logged-in user.
    """

    user_id = str(
        current_user["_id"]
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    total_uploads = (
        uploads_collection.count_documents(
            {
                "user_id": user_id
            }
        )
    )


    genuine_audio = (
        uploads_collection.count_documents(
            {
                "user_id": user_id,
                "prediction": "Genuine",
            }
        )
    )


    tts_audio = (
        uploads_collection.count_documents(
            {
                "user_id": user_id,
                "prediction": "TTS",
            }
        )
    )


    voice_conversion_audio = (
        uploads_collection.count_documents(
            {
                "user_id": user_id,
                "prediction": "Voice Conversion",
            }
        )
    )


    # ========================================================
    # RECENT ACTIVITY
    # ========================================================

    recent_uploads = (
        uploads_collection
        .find(
            {
                "user_id": user_id
            }
        )
        .sort(
            "uploaded_at",
            -1
        )
        .limit(3)
    )


    recent_activity = []


    for upload in recent_uploads:

        original_filename = upload.get(
            "original_filename",
            "Unknown file"
        )


        status = upload.get(
            "status",
            "uploaded"
        )


        prediction = upload.get(
            "prediction"
        )


        confidence = upload.get(
            "confidence"
        )


        uploaded_at = upload.get(
            "uploaded_at"
        )


        # ----------------------------------------------------
        # Build activity description
        # ----------------------------------------------------

        if (
            status == "completed"
            and prediction
        ):

            if confidence is not None:

                description = (
                    f"{prediction} "
                    f"• {confidence:.2f}% confidence"
                )

            else:

                description = prediction

        elif status == "uploaded":

            description = (
                "Waiting for analysis"
            )

        elif status == "processing":

            description = (
                "Analysis in progress"
            )

        elif status == "failed":

            description = (
                "Analysis failed"
            )

        else:

            description = (
                status.capitalize()
            )


        # ----------------------------------------------------
        # Convert timestamp
        # ----------------------------------------------------

        if uploaded_at:

            timestamp = (
                uploaded_at
                .replace(
                    tzinfo=timezone.utc
                )
                .isoformat()
            )

        else:

            timestamp = ""


        # ----------------------------------------------------
        # Add activity
        # ----------------------------------------------------

        recent_activity.append(

            RecentActivity(

                title=original_filename,

                description=description,

                timestamp=timestamp,

                prediction=prediction,

                confidence=(
                    float(confidence)
                    if confidence is not None
                    else None
                ),

                status=status,

            )

        )


    # ========================================================
    # RETURN DASHBOARD
    # ========================================================

    return DashboardResponse(

        full_name=current_user[
            "full_name"
        ],

        email=current_user[
            "email"
        ],

        statistics=DashboardStatistics(

            total_uploads=total_uploads,

            genuine_audio=genuine_audio,

            tts_audio=tts_audio,

            voice_conversion_audio=(
                voice_conversion_audio
            ),

        ),

        recent_activity=recent_activity,

    )