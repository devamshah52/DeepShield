from pymongo import MongoClient

from app.config import (
    MONGODB_URL,
    DATABASE_NAME,
)


# =====================================
# MongoDB Connection
# =====================================

client = MongoClient(MONGODB_URL)

db = client[DATABASE_NAME]


# =====================================
# Test MongoDB Connection
# =====================================

try:

    client.admin.command("ping")

    print("✅ Connected to MongoDB successfully!")

except Exception as e:

    print(
        "❌ MongoDB connection failed:",
        e
    )


# =====================================
# Collections
# =====================================

# Normal users
users_collection = db["users"]


# Audio uploads / detection records
uploads_collection = db["uploads"]


# Admin accounts
admins_collection = db["admins"]