from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analytics
from app.routers.auth import router as auth_router
from app.routers.home import router as home_router
from app.routers.user import router as user_router
from app.routers.upload import router as upload_router
from app.routers import dashboard
from app.routers import history
from app.routers import admin


app = FastAPI(
    title="DeepShield API",
    description="Backend API for the DeepShield Deepfake Audio and Voice Clone Detection System.",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,

    allow_origins=origins,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(home_router)

app.include_router(user_router)

app.include_router(auth_router)

app.include_router(upload_router)

app.include_router(dashboard.router)

app.include_router(history.router)

app.include_router(analytics.router)

# Admin routes
app.include_router(admin.router)