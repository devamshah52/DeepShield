from fastapi import APIRouter
from app.database.mongodb import users_collection
from app.models.user import User

router = APIRouter()


@router.post("/user")
def create_user(user: User):
    result = users_collection.insert_one(user.model_dump())

    return {
        "message": "User created successfully",
        "id": str(result.inserted_id)
    }


@router.get("/users")
def get_users():
    users = []

    for user in users_collection.find():
        user["_id"] = str(user["_id"])
        users.append(user)

    return users


@router.get("/user/{email}")
def get_user(email: str):
    user = users_collection.find_one({"email": email})

    if user is None:
        return {"message": "User not found"}

    user["_id"] = str(user["_id"])

    return user


@router.put("/user/{email}")
def update_user(email: str, user: User):
    result = users_collection.update_one(
        {"email": email},
        {"$set": user.model_dump()}
    )

    if result.matched_count == 0:
        return {"message": "User not found"}

    return {"message": "User updated successfully"}


@router.delete("/user/{email}")
def delete_user(email: str):
    result = users_collection.delete_one({"email": email})

    if result.deleted_count == 0:
        return {"message": "User not found"}

    return {"message": "User deleted successfully"}