from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
)

from fastapi.security import OAuth2PasswordRequestForm

from bson import ObjectId

from app.database.mongodb import (
    admins_collection,
    users_collection,
    uploads_collection,
)

from app.models.admin import (
    AdminRegister,
)

from app.utils.security import (
    hash_password,
    verify_password,
)

from app.utils.jwt_handler import (
    create_access_token,
    get_current_admin,
)


router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)


# ============================================================
# ADMIN REGISTRATION
# ============================================================

@router.post("/register")
def register_admin(
    admin: AdminRegister
):

    # --------------------------------------------------------
    # Normalize email
    # --------------------------------------------------------

    email = admin.email.strip().lower()


    # --------------------------------------------------------
    # Check whether an admin already exists
    # --------------------------------------------------------

    existing_admin_count = (
        admins_collection.count_documents({})
    )


    if existing_admin_count > 0:

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Admin account already exists. "
                "Registration is disabled."
            ),
        )


    # --------------------------------------------------------
    # Extra email check
    # --------------------------------------------------------

    existing_admin = admins_collection.find_one(
        {
            "email": email
        }
    )


    if existing_admin:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Admin email already registered.",
        )


    # --------------------------------------------------------
    # Hash password
    # --------------------------------------------------------

    hashed_password = hash_password(
        admin.password
    )


    # --------------------------------------------------------
    # Create admin document
    # --------------------------------------------------------

    admin_data = {

        "full_name": admin.full_name.strip(),

        "email": email,

        "password": hashed_password,

        "role": "admin",

    }


    # --------------------------------------------------------
    # Insert admin
    # --------------------------------------------------------

    result = admins_collection.insert_one(
        admin_data
    )


    return {

        "message": "Admin registration successful.",

        "id": str(result.inserted_id),

        "email": email,

    }


# ============================================================
# ADMIN LOGIN
# ============================================================

@router.post("/login")
def login_admin(
    form_data: OAuth2PasswordRequestForm = Depends()
):

    # --------------------------------------------------------
    # Normalize email
    # --------------------------------------------------------

    email = form_data.username.strip().lower()


    # --------------------------------------------------------
    # Find admin
    # --------------------------------------------------------

    admin = admins_collection.find_one(
        {
            "email": email
        }
    )


    if admin is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin email or password.",
        )


    # --------------------------------------------------------
    # Verify password
    # --------------------------------------------------------

    if not verify_password(
        form_data.password,
        admin["password"]
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin email or password.",
        )


    # --------------------------------------------------------
    # Create Admin JWT
    # --------------------------------------------------------

    access_token = create_access_token(
        {
            "email": admin["email"],
            "role": "admin",
        }
    )


    return {

        "access_token": access_token,

        "token_type": "bearer",

    }


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@router.get("/dashboard")
async def admin_dashboard(
    current_admin: dict = Depends(
        get_current_admin
    )
):

    # ========================================================
    # USER STATISTICS
    # ========================================================

    total_users = users_collection.count_documents({})


    # ========================================================
    # UPLOAD STATISTICS
    # ========================================================

    total_uploads = (
        uploads_collection.count_documents({})
    )


    completed_uploads = (
        uploads_collection.count_documents(
            {
                "status": "completed"
            }
        )
    )


    failed_uploads = (
        uploads_collection.count_documents(
            {
                "status": "failed"
            }
        )
    )


    # ========================================================
    # PREDICTION STATISTICS
    # ========================================================

    genuine_audio = (
        uploads_collection.count_documents(
            {
                "prediction": "Genuine"
            }
        )
    )


    deepfake_audio = (
        uploads_collection.count_documents(
            {
                "prediction": "Deepfake"
            }
        )
    )


    voice_clone_audio = (
        uploads_collection.count_documents(
            {
                "prediction": "Voice Clone"
            }
        )
    )


    # ========================================================
    # RECENT ACTIVITY
    # ========================================================

    recent_uploads = list(
        uploads_collection.find(
            {},
            {
                "user_id": 1,
                "original_filename": 1,
                "prediction": 1,
                "confidence": 1,
                "status": 1,
                "uploaded_at": 1,
            },
        )
        .sort(
            "uploaded_at",
            -1
        )
        .limit(10)
    )


    # ========================================================
    # FORMAT RECENT ACTIVITY
    # ========================================================

    recent_activity = []


    for upload in recent_uploads:

        # ----------------------------------------------------
        # Get user information
        # ----------------------------------------------------

        user_id = upload.get(
            "user_id"
        )


        user_name = "Unknown User"


        if user_id:

            try:

                user = users_collection.find_one(
                    {
                        "_id": ObjectId(user_id)
                    },
                    {
                        "full_name": 1,
                        "email": 1,
                    }
                )


                if user:

                    user_name = user.get(
                        "full_name",
                        "Unknown User"
                    )


            except Exception:

                user_name = "Unknown User"


        # ----------------------------------------------------
        # Add activity
        # ----------------------------------------------------

        # ----------------------------------------------------
        # IMPORTANT:
        # MongoDB stores the upload datetime in UTC.
        #
        # The frontend uses:
        # new Date(activity.uploaded_at)
        #
        # followed by:
        # .toLocaleString()
        #
        # Therefore we explicitly mark a naive MongoDB
        # datetime as UTC by adding "Z".
        #
        # This allows the browser to correctly convert
        # the timestamp to the user's local timezone.
        # ----------------------------------------------------

        uploaded_at = upload.get(
            "uploaded_at"
        )


        if uploaded_at:

            uploaded_at_string = (
                uploaded_at.isoformat()
            )


            if (
                not uploaded_at_string.endswith("Z")
                and "+" not in uploaded_at_string
            ):

                uploaded_at_string += "Z"

        else:

            uploaded_at_string = None


        recent_activity.append(
            {

                "id": str(
                    upload["_id"]
                ),

                "user_id": user_id,

                "user_name": user_name,

                "filename": upload.get(
                    "original_filename"
                ),

                "prediction": upload.get(
                    "prediction"
                ),

                "confidence": upload.get(
                    "confidence"
                ),

                "status": upload.get(
                    "status"
                ),

                "uploaded_at": uploaded_at_string,

            }
        )


    # ========================================================
    # ADMIN INFORMATION
    # ========================================================

    return {

        "admin": {

            "full_name": current_admin.get(
                "full_name"
            ),

            "email": current_admin.get(
                "email"
            ),

            "role": current_admin.get(
                "role"
            ),

        },


        "statistics": {

            "total_users": total_users,

            "total_uploads": total_uploads,

            "completed_uploads": (
                completed_uploads
            ),

            "failed_uploads": (
                failed_uploads
            ),

            "genuine_audio": genuine_audio,

            "deepfake_audio": deepfake_audio,

            "voice_clone_audio": (
                voice_clone_audio
            ),

        },


        "recent_activity": recent_activity,

    }


# ============================================================
# ADMIN - GET ALL USERS
# ============================================================

@router.get("/users")
async def get_all_users(

    current_admin: dict = Depends(
        get_current_admin
    ),

    search: str = Query(
        default="",
        description="Search users by name or email."
    ),

):

    # ========================================================
    # SEARCH FILTER
    # ========================================================

    search = search.strip()


    if search:

        search_filter = {

            "$or": [

                {
                    "full_name": {
                        "$regex": search,
                        "$options": "i"
                    }
                },

                {
                    "email": {
                        "$regex": search,
                        "$options": "i"
                    }
                },

            ]

        }

    else:

        search_filter = {}


    # ========================================================
    # FETCH USERS
    # ========================================================

    users = list(
        users_collection.find(
            search_filter,
            {
                "password": 0
            }
        ).sort(
            "_id",
            -1
        )
    )


    # ========================================================
    # FORMAT USERS
    # ========================================================

    user_list = []


    for user in users:

        user_id = str(
            user["_id"]
        )


        # ----------------------------------------------------
        # Count uploads
        # ----------------------------------------------------

        total_uploads = (
            uploads_collection.count_documents(
                {
                    "user_id": user_id
                }
            )
        )


        # ----------------------------------------------------
        # Registration date
        # ----------------------------------------------------

        registered_at = user.get(
            "created_at"
        )


        if registered_at:

            try:

                registered_at = (
                    registered_at.isoformat()
                )

            except AttributeError:

                registered_at = str(
                    registered_at
                )


        # ----------------------------------------------------
        # User response
        # ----------------------------------------------------

        user_list.append(
            {

                "id": user_id,

                "full_name": user.get(
                    "full_name",
                    "Unknown User"
                ),

                "email": user.get(
                    "email",
                    ""
                ),

                "role": user.get(
                    "role",
                    "user"
                ),

                "registered_at": (
                    registered_at
                ),

                "total_uploads": (
                    total_uploads
                ),

                "is_blocked": user.get(
                    "is_blocked",
                    False
                ),

            }
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "users": user_list,

        "total_users": len(
            user_list
        ),

    }


# ============================================================
# ADMIN - BLOCK USER
# ============================================================

@router.patch("/users/{user_id}/block")
async def block_user(

    user_id: str,

    current_admin: dict = Depends(
        get_current_admin
    ),

):

    # ========================================================
    # VALIDATE USER ID
    # ========================================================

    if not ObjectId.is_valid(user_id):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID.",
        )


    # ========================================================
    # FIND USER
    # ========================================================

    user = users_collection.find_one(
        {
            "_id": ObjectId(user_id)
        }
    )


    if user is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )


    # ========================================================
    # PREVENT ADMIN ACCOUNT FROM BEING BLOCKED
    # ========================================================

    if user.get("role") == "admin":

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin accounts cannot be blocked.",
        )


    # ========================================================
    # BLOCK USER
    # ========================================================

    users_collection.update_one(
        {
            "_id": ObjectId(user_id)
        },
        {
            "$set": {
                "is_blocked": True
            }
        }
    )


    return {

        "message": "User blocked successfully.",

        "user_id": user_id,

        "is_blocked": True,

    }


# ============================================================
# ADMIN - UNBLOCK USER
# ============================================================

@router.patch("/users/{user_id}/unblock")
async def unblock_user(

    user_id: str,

    current_admin: dict = Depends(
        get_current_admin
    ),

):

    # ========================================================
    # VALIDATE USER ID
    # ========================================================

    if not ObjectId.is_valid(user_id):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID.",
        )


    # ========================================================
    # FIND USER
    # ========================================================

    user = users_collection.find_one(
        {
            "_id": ObjectId(user_id)
        }
    )


    if user is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )


    # ========================================================
    # PREVENT ADMIN ACCOUNT FROM BEING UNBLOCKED
    # ========================================================

    if user.get("role") == "admin":

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin accounts cannot be modified.",
        )


    # ========================================================
    # UNBLOCK USER
    # ========================================================

    users_collection.update_one(
        {
            "_id": ObjectId(user_id)
        },
        {
            "$set": {
                "is_blocked": False
            }
        }
    )


    return {

        "message": "User unblocked successfully.",

        "user_id": user_id,

        "is_blocked": False,

    }


# ============================================================
# ADMIN - DELETE USER
# ============================================================

@router.delete("/users/{user_id}")
async def delete_user(

    user_id: str,

    current_admin: dict = Depends(
        get_current_admin
    ),

):

    # ========================================================
    # VALIDATE USER ID
    # ========================================================

    if not ObjectId.is_valid(user_id):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID.",
        )


    # ========================================================
    # FIND USER
    # ========================================================

    user = users_collection.find_one(
        {
            "_id": ObjectId(user_id)
        }
    )


    if user is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )


    # ========================================================
    # PREVENT ADMIN ACCOUNT DELETION
    # ========================================================

    if user.get("role") == "admin":

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin accounts cannot be deleted.",
        )


    # ========================================================
    # DELETE USER UPLOAD RECORDS
    # ========================================================

    uploads_collection.delete_many(
        {
            "user_id": user_id
        }
    )


    # ========================================================
    # DELETE USER
    # ========================================================

    result = users_collection.delete_one(
        {
            "_id": ObjectId(user_id)
        }
    )


    if result.deleted_count == 0:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User could not be deleted.",
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "message": "User deleted successfully.",

        "user_id": user_id,

        "email": user.get(
            "email"
        ),

    }