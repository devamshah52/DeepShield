from pydantic import BaseModel, EmailStr


class AdminRegister(BaseModel):

    full_name: str

    email: EmailStr

    password: str


class AdminLogin(BaseModel):

    email: EmailStr

    password: str