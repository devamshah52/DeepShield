from pydantic import BaseModel
from typing import List, Optional


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

class DashboardStatistics(BaseModel):

    total_uploads: int

    genuine_audio: int

    tts_audio: int

    voice_conversion_audio: int


# ============================================================
# RECENT ACTIVITY
# ============================================================

class RecentActivity(BaseModel):

    title: str

    description: str

    timestamp: str

    prediction: Optional[str] = None

    confidence: Optional[float] = None

    status: Optional[str] = None


# ============================================================
# DASHBOARD RESPONSE
# ============================================================

class DashboardResponse(BaseModel):

    full_name: str

    email: str

    statistics: DashboardStatistics

    recent_activity: List[RecentActivity]