from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.database.mongodb import users_collection

from app.models.register import RegisterUser

from app.models.forgot_password import ForgotPasswordRequest
from app.models.verify_otp import VerifyOTPRequest
from app.models.reset_password import ResetPasswordRequest

from app.utils.email_sender import send_otp_email
from app.utils.otp import generate_otp

from app.models.register_verify import RegisterVerifyRequest

from app.utils.otp_storage import (
    save_otp,
    verify_otp,
    remove_otp,
    get_extra_data,
)

from app.utils.security import (
    hash_password,
    verify_password,
    oauth2_scheme,
)

from app.utils.jwt_handler import (
    create_access_token,
    verify_access_token,
)


router = APIRouter()


# =====================================
# Send Registration OTP
# =====================================

@router.post("/register/send-otp")
def send_registration_otp(
    user: RegisterUser
):

    email = user.email.strip().lower()

    existing_user = users_collection.find_one(
        {
            "email": email
        }
    )

    if existing_user:

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered."
        )

    otp = generate_otp()

    save_otp(

        email,

        otp,

        {

            "full_name": user.full_name,

            "email": email,

            "password": user.password

        }

    )

    email_sent = send_otp_email(
        email,
        otp
    )

    if not email_sent:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP."
        )

    return {

        "message": "OTP sent successfully."

    }


# =====================================
# Verify Registration OTP
# =====================================

@router.post("/register/verify-otp")
def verify_registration_otp(
    request: RegisterVerifyRequest
):

    email = request.email.strip().lower()

    if not verify_otp(
        email,
        request.otp
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP."
        )

    data = get_extra_data(email)

    if data is None:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration data not found."
        )

    existing_user = users_collection.find_one(
        {
            "email": email
        }
    )

    if existing_user:

        remove_otp(email)

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered."
        )

    data["password"] = hash_password(
        data["password"]
    )

    result = users_collection.insert_one(
        data
    )

    remove_otp(email)

    return {

        "message": "Registration successful.",

        "id": str(result.inserted_id)

    }


# =====================================
# Register
# =====================================

# @router.post("/register", status_code=status.HTTP_201_CREATED)
# def register(user: RegisterUser):
#
#     email = user.email.strip().lower()
#
#     existing_user = users_collection.find_one(
#         {"email": email}
#     )
#
#     if existing_user:
#
#         raise HTTPException(
#             status_code=status.HTTP_409_CONFLICT,
#             detail="Email already registered."
#         )
#
#     user_data = user.model_dump()
#
#     user_data["email"] = email
#
#     user_data["password"] = hash_password(
#         user.password
#     )
#
#     result = users_collection.insert_one(
#         user_data
#     )
#
#     return {
#
#         "message": "Registration successful",
#
#         "id": str(result.inserted_id),
#
#         "email": email
#
#     }


# =====================================
# Login
# =====================================

@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):

    email = form_data.username.strip().lower()

    existing_user = users_collection.find_one(
        {
            "email": email
        }
    )

    if existing_user is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # =====================================
    # Verify Password
    # =====================================

    if not verify_password(
        form_data.password,
        existing_user["password"]
    ):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # =====================================
    # Check Whether User Is Blocked
    # =====================================

    if existing_user.get("is_blocked", False):

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been blocked by the administrator."
        )

    # =====================================
    # Create Access Token
    # =====================================

    access_token = create_access_token(
        {
            "email": existing_user["email"]
        }
    )

    return {

        "access_token": access_token,

        "token_type": "bearer"

    }


# =====================================
# Profile
# =====================================

@router.get("/profile")
def profile(
    token: str = Depends(oauth2_scheme)
):

    payload = verify_access_token(token)

    if payload is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token."
        )

    return {

        "message": "Profile accessed successfully",

        "user": payload

    }


# =====================================
# Forgot Password
# =====================================

@router.post("/forgot-password")
def forgot_password(
    request: ForgotPasswordRequest
):

    email = request.email.strip().lower()

    existing_user = users_collection.find_one(
        {
            "email": email
        }
    )

    if existing_user is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No account found with this email."
        )

    otp = generate_otp()

    save_otp(
        email,
        otp
    )

    email_sent = send_otp_email(
        email,
        otp
    )

    if not email_sent:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP email."
        )

    return {

        "message": "OTP sent successfully."

    }


# =====================================
# Verify OTP
# =====================================

@router.post("/verify-otp")
def verify_otp_route(
    request: VerifyOTPRequest
):

    email = request.email.strip().lower()

    is_valid = verify_otp(
        email,
        request.otp
    )

    if not is_valid:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP."
        )

    return {

        "message": "OTP verified successfully."

    }


# =====================================
# Reset Password
# =====================================

@router.post("/reset-password")
def reset_password(
    request: ResetPasswordRequest
):

    email = request.email.strip().lower()

    if not verify_otp(
        email,
        request.otp
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP."
        )

    hashed_password = hash_password(
        request.new_password
    )

    users_collection.update_one(
        {
            "email": email
        },
        {
            "$set": {
                "password": hashed_password
            }
        }
    )

    remove_otp(email)

    return {

        "message": "Password reset successfully."

    }