from pydantic import BaseModel, EmailStr


class RegisterVerifyRequest(BaseModel):

    email: EmailStr

    otp: str