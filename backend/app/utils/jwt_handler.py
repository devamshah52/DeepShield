from datetime import datetime, timedelta

from jose import jwt, JWTError

from fastapi import (
    Depends,
    HTTPException,
    status,
)

from app.config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

from app.utils.security import (
    oauth2_scheme,
    admin_oauth2_scheme,
)

from app.database.mongodb import (
    users_collection,
    admins_collection,
)


# ============================================================
# CREATE ACCESS TOKEN
# ============================================================

def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# ============================================================
# VERIFY ACCESS TOKEN
# ============================================================

def verify_access_token(token: str):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return payload

    except JWTError:

        return None


# ============================================================
# GET CURRENT USER
# ============================================================

async def get_current_user(
    token: str = Depends(oauth2_scheme)
):

    payload = verify_access_token(token)

    if payload is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token."
        )


    email = payload.get("email")

    if email is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token."
        )


    # --------------------------------------------------------
    # Prevent Admin token from being used as User token
    # --------------------------------------------------------

    if payload.get("role") == "admin":

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Admin access token cannot be used "
                "for user routes."
            )
        )


    # --------------------------------------------------------
    # Find User
    # --------------------------------------------------------

    user = users_collection.find_one(
        {
            "email": email
        },
        {
            "password": 0
        }
    )


    if user is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )


    return user


# ============================================================
# GET CURRENT ADMIN
# ============================================================

async def get_current_admin(
    token: str = Depends(admin_oauth2_scheme)
):

    payload = verify_access_token(token)

    if payload is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired admin token."
        )


    email = payload.get("email")

    role = payload.get("role")


    # --------------------------------------------------------
    # Make sure this is an Admin token
    # --------------------------------------------------------

    if email is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token."
        )


    if role != "admin":

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )


    # --------------------------------------------------------
    # Find Admin
    # --------------------------------------------------------

    admin = admins_collection.find_one(
        {
            "email": email
        },
        {
            "password": 0
        }
    )


    if admin is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found."
        )


    return admin