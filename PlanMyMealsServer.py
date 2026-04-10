import asyncio
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app_json_async import init_app_storage_sync, reset_app_documents_sync
from db import crud
from db.session import get_db
from main import PlanMyMealsSystem


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = "default_session"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    agent_used: Optional[str] = None
    conversation_complete: Optional[bool] = False


class SystemStatus(BaseModel):
    status: str
    agents_loaded: List[str]
    databases_loaded: List[str]
    api_key_configured: bool
    uptime: str


class UserProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = None
    goal_weight: Optional[float] = None
    goal_type: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = []
    allergies: Optional[List[str]] = []


class MealEntry(BaseModel):
    food_name: str
    meal_type: str  # breakfast | lunch | dinner | snacks
    date: Optional[str] = None  # YYYY-MM-DD
    servings: Optional[float] = 1.0
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fats: Optional[float] = None


class ProgressResponse(BaseModel):
    date: str
    daily_totals: Dict[str, float]
    daily_percentages: Optional[Dict[str, float]] = None
    meal_breakdown: Dict[str, Dict[str, float]]
    targets: Optional[Dict[str, float]] = None


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlanMyMeals API",
    description="AI-powered nutrition assistant with meal planning and tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = datetime.now()
nutrify_system: Optional[PlanMyMealsSystem] = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global nutrify_system
    load_dotenv()
    init_app_storage_sync()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required but was not found in the environment.")

    try:
        nutrify_system = PlanMyMealsSystem(api_key)
        print("PlanMyMeals FastAPI server started successfully!")
    except Exception as e:
        print(f"Failed to initialise PlanMyMeals: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    print("PlanMyMeals FastAPI server shutting down...")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if not nutrify_system:
                await websocket.send_text("System not initialised.")
                continue
            response = await nutrify_system.process_message_async(data)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")


# ── Health & status ───────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - start_time),
    }


@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    if not nutrify_system:
        raise HTTPException(status_code=503, detail="System not initialised")

    agents_loaded = [
        name
        for name, attr in [
            ("conversation", "conversation_agent"),
            ("manager", "manager_agent"),
            ("meal_plan", "meal_plan_agent"),
            ("meal_track", "meal_track_agent"),
        ]
        if hasattr(nutrify_system, attr)
    ]

    databases_loaded = [
        label
        for label, path in [
            ("calorie_library", "calorie_library.csv"),
            ("indian_recipes", "indian_recipes.csv"),
        ]
        if os.path.exists(path)
    ]

    return SystemStatus(
        status="operational",
        agents_loaded=agents_loaded,
        databases_loaded=databases_loaded,
        api_key_configured=bool(nutrify_system.api_key),
        uptime=str(datetime.now() - start_time),
    )


# ── Profile ───────────────────────────────────────────────────────────────────

_PROFILE_FIELDS = {"name", "age", "weight", "height", "gender", "activity_level"}
_GOAL_FIELDS = {"goal_weight", "goal_type"}
_PREF_FIELDS = {"dietary_restrictions", "allergies"}


@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    user_data = crud.get_document(db, user_id, "user_info")
    if user_data is None:
        raise HTTPException(status_code=404, detail="No profile found for this user")
    return user_data


@app.put("/profile/{user_id}")
async def update_user_profile(
    user_id: str, profile: UserProfile, db: Session = Depends(get_db)
):
    user_data = crud.get_document(db, user_id, "user_info") or {}
    user_data.setdefault("profile", {})
    user_data.setdefault("goals", {})
    user_data.setdefault("preferences", {})

    for field, value in profile.model_dump(exclude_unset=True).items():
        if field in _PROFILE_FIELDS:
            user_data["profile"][field] = value
        elif field in _GOAL_FIELDS:
            user_data["goals"][field] = value
        elif field in _PREF_FIELDS:
            user_data["preferences"][field] = value

    crud.set_document(db, user_id, "user_info", user_data)
    return {"message": "Profile updated successfully", "profile": user_data}


# ── Meal tracking ─────────────────────────────────────────────────────────────

@app.post("/meals/log")
async def log_meal(meal: MealEntry):
    if not nutrify_system:
        raise HTTPException(status_code=503, detail="System not initialised")

    parts = [f"I had {meal.food_name} for {meal.meal_type}"]
    if meal.date:
        parts.append(f"on {meal.date}")
    if meal.servings and meal.servings != 1:
        parts.append(f"{meal.servings} servings")
    if meal.calories:
        parts.append(f"with {meal.calories} calories")
    if meal.protein:
        parts.append(f"{meal.protein}g protein")
    if meal.carbs:
        parts.append(f"{meal.carbs}g carbs")
    if meal.fats:
        parts.append(f"{meal.fats}g fats")

    meal_text = " ".join(parts)
    response = await nutrify_system.process_message_async(meal_text)

    return {
        "message": "Meal logged successfully",
        "agent_response": response,
        "logged_meal": meal.model_dump(),
    }


@app.get("/meals/progress")
async def get_daily_progress(
    date: Optional[str] = None,
    user_id: Optional[str] = "default",
    db: Session = Depends(get_db),
):
    meal_log = crud.get_document(db, user_id, "meal_log")
    if not meal_log:
        raise HTTPException(status_code=404, detail="No meal data found for this user")

    target_date = date or str(date.today())
    summary = meal_log.get("daily_summaries", {}).get(target_date)
    if not summary:
        raise HTTPException(status_code=404, detail=f"No data found for {target_date}")

    return ProgressResponse(
        date=target_date,
        daily_totals={
            "calories": summary.get("total_calories", 0),
            "protein": summary.get("total_protein", 0),
            "carbs": summary.get("total_carbs", 0),
            "fats": summary.get("total_fats", 0),
        },
        daily_percentages=summary.get("daily_percentages"),
        meal_breakdown=summary.get("meal_breakdown", {}),
        targets=summary.get("targets"),
    )


@app.get("/meals/history")
async def get_meal_history(
    days: int = 7,
    user_id: Optional[str] = "default",
    db: Session = Depends(get_db),
):
    meal_log = crud.get_document(db, user_id, "meal_log")
    if not meal_log:
        raise HTTPException(status_code=404, detail="No meal data found for this user")

    meal_entries = meal_log.get("meal_entries", {})
    daily_summaries = meal_log.get("daily_summaries", {})
    recent_dates = sorted(meal_entries.keys(), reverse=True)[:days]

    history = [
        {
            "date": d,
            "meals": meal_entries[d],
            "daily_totals": {
                "calories": daily_summaries.get(d, {}).get("total_calories", 0),
                "protein": daily_summaries.get(d, {}).get("total_protein", 0),
                "carbs": daily_summaries.get(d, {}).get("total_carbs", 0),
                "fats": daily_summaries.get(d, {}).get("total_fats", 0),
            },
            "meal_count": sum(len(v) for v in meal_entries[d].values()),
        }
        for d in recent_dates
    ]

    return {"history": history, "total_days": len(history)}


# ── Meal planning ─────────────────────────────────────────────────────────────

@app.post("/meal-plan/generate")
async def generate_meal_plan(user_id: Optional[str] = "default"):
    if not nutrify_system:
        raise HTTPException(status_code=503, detail="System not initialised")

    response = await nutrify_system.process_message_async("create meal plan")
    return {"message": "Meal plan generated", "meal_plan": response}


# ── Food search ───────────────────────────────────────────────────────────────

@app.get("/foods/search")
async def search_foods(query: str, limit: int = 10):
    import pandas as pd

    foods: List[Dict[str, Any]] = []
    half = max(limit // 2, 1)

    if os.path.exists("calorie_library.csv"):
        df = pd.read_csv("calorie_library.csv")
        for _, row in df[df["Food Item"].str.contains(query, case=False, na=False)].head(half).iterrows():
            foods.append({
                "name": row.get("Food Item"),
                "calories": float(row.get("Calories (Cal)", 0)),
                "protein": float(row.get("Protein (g)", 0)),
                "carbs": float(row.get("Carbs (g)", 0)),
                "fats": float(row.get("Fats (g)", 0)),
                "serving_size": row.get("Serving Size"),
                "source": "calorie_library",
            })

    if os.path.exists("indian_recipes.csv"):
        df = pd.read_csv("indian_recipes.csv")
        for _, row in df[df["Recipe Name"].str.contains(query, case=False, na=False)].head(half).iterrows():
            foods.append({
                "name": row.get("Recipe Name"),
                "calories": float(row.get("Calories (per serving)", 0)),
                "protein": float(row.get("Protein (g)", 0)),
                "carbs": float(row.get("Carbs (g)", 0)),
                "fats": float(row.get("Fats (g)", 0)),
                "ingredients": row.get("Ingredients"),
                "source": "indian_recipes",
            })

    return {"query": query, "results": foods[:limit], "total_found": len(foods)}


# ── Data reset ────────────────────────────────────────────────────────────────

@app.delete("/reset/{user_id}")
async def reset_user_data(user_id: str, confirm: bool = False):
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to confirm the reset")
    reset_app_documents_sync(user_id)
    return {"message": f"All data for user '{user_id}' has been reset"}


# ── Static files (must be last so it doesn't shadow API routes) ───────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "PlanMyMealsServer:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
