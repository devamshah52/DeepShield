from passlib.context import CryptContext

from fastapi.security import OAuth2PasswordBearer


# ============================================================
# PASSWORD HASHING
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password: str):

    return pwd_context.hash(
        password
    )


def verify_password(
    plain_password: str,
    hashed_password: str
):

    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# ============================================================
# NORMAL USER AUTHENTICATION
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="login",
    scheme_name="UserAuthentication"
)


# ============================================================
# ADMIN AUTHENTICATION
# ============================================================

admin_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="admin/login",
    scheme_name="AdminAuthentication"
)