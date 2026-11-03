from collections import defaultdict
from datetime import timezone

from fastapi import APIRouter, Depends

from app.database.mongodb import uploads_collection
from app.utils.jwt_handler import get_current_user


router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)


# ============================================================
# GET USER DETECTION ANALYTICS
# ============================================================

@router.get("/")
async def get_analytics(
    current_user: dict = Depends(get_current_user),
):
    """
    Return detection analytics for the currently
    logged-in user.
    """

    user_id = str(
        current_user["_id"]
    )

    # ========================================================
    # FETCH USER RECORDS
    # ========================================================

    uploads = list(
        uploads_collection.find(
            {
                "user_id": user_id
            }
        ).sort(
            "uploaded_at",
            1
        )
    )

    # ========================================================
    # INITIAL VALUES
    # ========================================================

    total_uploads = len(uploads)

    completed = 0
    failed = 0
    pending = 0

    prediction_counts = {
        "Genuine": 0,
        "TTS": 0,
        "Voice Conversion": 0,
    }

    confidence_values = []

    confidence_by_prediction = defaultdict(list)

    format_counts = defaultdict(int)

    detections_by_date = defaultdict(int)

    # ========================================================
    # PROCESS USER RECORDS
    # ========================================================

    for upload in uploads:

        status = upload.get(
            "status"
        )

        prediction = upload.get(
            "prediction"
        )

        confidence = upload.get(
            "confidence"
        )

        original_filename = upload.get(
            "original_filename",
            "",
        )

        uploaded_at = upload.get(
            "uploaded_at"
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if status == "completed":
            completed += 1

        elif status == "failed":
            failed += 1

        else:
            pending += 1

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if prediction in prediction_counts:

            prediction_counts[
                prediction
            ] += 1

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence is not None:

            try:

                confidence_float = float(
                    confidence
                )

                confidence_values.append(
                    confidence_float
                )

                if prediction in prediction_counts:

                    confidence_by_prediction[
                        prediction
                    ].append(
                        confidence_float
                    )

            except (
                TypeError,
                ValueError,
            ):

                pass

        # ----------------------------------------------------
        # FILE FORMAT
        # ----------------------------------------------------

        if original_filename:

            filename_lower = (
                original_filename.lower()
            )

            if "." in filename_lower:

                extension = (
                    filename_lower
                    .rsplit(".", 1)[-1]
                    .upper()
                )

                format_counts[
                    extension
                ] += 1

        # ----------------------------------------------------
        # DETECTIONS BY DATE
        # ----------------------------------------------------

        if uploaded_at:

            try:

                if uploaded_at.tzinfo is None:

                    uploaded_at = uploaded_at.replace(
                        tzinfo=timezone.utc
                    )

                date_key = (
                    uploaded_at
                    .astimezone(timezone.utc)
                    .strftime("%Y-%m-%d")
                )

                detections_by_date[
                    date_key
                ] += 1

            except (
                AttributeError,
                ValueError,
            ):

                pass

    # ========================================================
    # AVERAGE CONFIDENCE
    # ========================================================

    if confidence_values:

        average_confidence = round(
            sum(confidence_values)
            / len(confidence_values),
            2,
        )

    else:

        average_confidence = 0.0

    # ========================================================
    # AVERAGE CONFIDENCE BY PREDICTION
    # ========================================================

    average_confidence_by_prediction = {}

    for prediction in prediction_counts:

        values = confidence_by_prediction.get(
            prediction,
            []
        )

        if values:

            average_confidence_by_prediction[
                prediction
            ] = round(
                sum(values)
                / len(values),
                2,
            )

        else:

            average_confidence_by_prediction[
                prediction
            ] = 0.0

    # ========================================================
    # TIME SERIES
    # ========================================================

    detection_timeline = [

        {
            "date": date,
            "count": count,
        }

        for date, count
        in sorted(
            detections_by_date.items()
        )

    ]

    # ========================================================
    # FORMAT DISTRIBUTION
    # ========================================================

    format_distribution = [

        {
            "format": audio_format,
            "count": count,
        }

        for audio_format, count
        in sorted(
            format_counts.items()
        )

    ]

    # ========================================================
    # RETURN ANALYTICS
    # ========================================================

    return {

        "summary": {

            "total_uploads":
                total_uploads,

            "completed":
                completed,

            "failed":
                failed,

            "pending":
                pending,

            "average_confidence":
                average_confidence,

        },

        "prediction_distribution": [

            {
                "prediction":
                    prediction,

                "count":
                    count,

            }

            for prediction, count
            in prediction_counts.items()

        ],

        "average_confidence_by_prediction": [

            {
                "prediction":
                    prediction,

                "confidence":
                    average_confidence_by_prediction[
                        prediction
                    ],

            }

            for prediction
            in prediction_counts

        ],

        "detection_timeline":
            detection_timeline,

        "format_distribution":
            format_distribution,

    }