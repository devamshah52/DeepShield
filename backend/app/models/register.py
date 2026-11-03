from pydantic import BaseModel, EmailStr, field_validator
import re


class RegisterUser(BaseModel):
    full_name: str
    email: EmailStr
    password: str

    @field_validator("full_name")
    @classmethod
    def validate_name(cls, value):

        value = value.strip()

        if len(value) < 3:
            raise ValueError(
                "Full name must contain at least 3 characters."
            )

        if len(value) > 50:
            raise ValueError(
                "Full name cannot exceed 50 characters."
            )

        if not re.fullmatch(r"[A-Za-z ]+", value):
            raise ValueError(
                "Full name can contain only letters and spaces."
            )

        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):

        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value):

        password_regex = (
            r"^(?=.*[a-z])"
            r"(?=.*[A-Z])"
            r"(?=.*\d)"
            r"(?=.*[@$!%*?&^#])"
            r"[A-Za-z\d@$!%*?&^#]{8,}$"
        )

        if not re.fullmatch(password_regex, value):
            raise ValueError(
                "Password must contain at least 8 characters, one uppercase letter, one lowercase letter, one number and one special character."
            )

        return value