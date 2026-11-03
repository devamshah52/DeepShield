from fastapi import APIRouter
from app.database.mongodb import users_collection

router = APIRouter()


@router.get("/")
def home():
    total_users = users_collection.count_documents({})

    return {
        "message": "Welcome to DeepShield Backend",
        "total_users": total_users
    }