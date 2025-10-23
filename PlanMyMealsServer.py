import asyncio
import os
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from main import PlanMyMealsSystem




# Pydantic models for API requests/responses
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
    meal_type: str  # breakfast, lunch, dinner, snacks
    date: Optional[str] = None  # YYYY-MM-DD format
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

# Initialize FastAPI app
app = FastAPI(
    title="PlanMyMeals API",
    description="AI-powered nutrition assistant with meal planning and tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


start_time = datetime.now()

# Session storage (use Redis or database in production)
active_sessions: Dict[str, Dict[str, Any]] = {}

PlanMyMeals_system: Optional[PlanMyMealsSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize PlanMyMeals system on startup"""
    global PlanMyMeals_system
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key is required")
        
        # Initialize the PlanMyMeals system
        PlanMyMeals_system = PlanMyMealsSystem(api_key)
        print("PlanMyMeals FastAPI server started successfully!")
        
    except Exception as e:
        print(f"Failed to initialize PlanMyMeals: {str(e)}")
        raise


# --- WebSocket Chat Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if not PlanMyMeals_system:
                await websocket.send_text("PlanMyMeals system not initialized.")
                continue
            response = await PlanMyMeals_system.process_message_async(data)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 PlanMyMeals FastAPI server shutting down...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": str(datetime.now() - start_time)
    }
app.mount("/", StaticFiles(directory="static", html=True), name="static")
# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get detailed system status"""
    if not PlanMyMeals_system:
        raise HTTPException(status_code=503, detail="PlanMyMeals system not initialized")
    
    # Check which agents are loaded
    agents_loaded = []
    if hasattr(PlanMyMeals_system, 'conversation_agent'):
        agents_loaded.append("conversation")
    if hasattr(PlanMyMeals_system, 'manager_agent'):
        agents_loaded.append("manager")
    if hasattr(PlanMyMeals_system, 'meal_plan_agent'):
        agents_loaded.append("meal_plan")
    if hasattr(PlanMyMeals_system, 'meal_track_agent'):
        agents_loaded.append("meal_track")
    
    # Check databases
    databases_loaded = []
    if os.path.exists("calorie_library.csv"):
        databases_loaded.append("calorie_library")
    if os.path.exists("indian_recipes.csv"):
        databases_loaded.append("indian_recipes")
    
    return SystemStatus(
        status="operational" if PlanMyMeals_system else "error",
        agents_loaded=agents_loaded,
        databases_loaded=databases_loaded,
        api_key_configured=bool(PlanMyMeals_system.api_key),
        uptime=str(datetime.now() - start_time)
    )


# User profile endpoints
@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile information"""
    try:
        if os.path.exists("user_info.json"):
            with open("user_info.json", "r") as f:
                user_data = json.load(f)
            return user_data
        else:
            return {"message": "No profile found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading profile: {str(e)}")

@app.put("/profile/{user_id}")
async def update_user_profile(user_id: str, profile: UserProfile):
    """Update user profile information"""
    try:
        # Load existing profile
        user_data = {}
        if os.path.exists("user_info.json"):
            with open("user_info.json", "r") as f:
                user_data = json.load(f)
        
        # Update profile section
        if "profile" not in user_data:
            user_data["profile"] = {}
        if "goals" not in user_data:
            user_data["goals"] = {}
        if "preferences" not in user_data:
            user_data["preferences"] = {}
        
        # Update fields that are provided
        profile_dict = profile.dict(exclude_unset=True)
        
        for field, value in profile_dict.items():
            if field in ["name", "age", "weight", "height", "gender", "activity_level"]:
                user_data["profile"][field] = value
            elif field in ["goal_weight", "goal_type"]:
                user_data["goals"][field] = value
            elif field in ["dietary_restrictions", "allergies"]:
                user_data["preferences"][field] = value
        
        # Save updated profile
        with open("user_info.json", "w") as f:
            json.dump(user_data, f, indent=2)
        
        return {"message": "Profile updated successfully", "profile": user_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")

# Meal tracking endpoints
@app.post("/meals/log")
async def log_meal(meal: MealEntry):
    """Log a meal entry"""
    if not PlanMyMeals_system:
        raise HTTPException(status_code=503, detail="PlanMyMeals system not initialized")
    
    try:
        # Convert meal entry to natural language for processing
        meal_text = f"I had {meal.food_name} for {meal.meal_type}"
        if meal.date:
            meal_text += f" on {meal.date}"
        if meal.servings and meal.servings != 1:
            meal_text += f", {meal.servings} servings"
        if meal.calories:
            meal_text += f" with {meal.calories} calories"
        if meal.protein:
            meal_text += f", {meal.protein}g protein"
        if meal.carbs:
            meal_text += f", {meal.carbs}g carbs"
        if meal.fats:
            meal_text += f", {meal.fats}g fats"
        
        # Process through PlanMyMeals system
        response = await PlanMyMeals_system.process_message_async(meal_text)
        
        return {
            "message": "Meal logged successfully",
            "PlanMyMeals_response": response,
            "logged_meal": meal.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging meal: {str(e)}")

@app.get("/meals/progress")
async def get_daily_progress(date: Optional[str] = None, user_id: Optional[str] = "default"):
    """Get daily nutrition progress"""
    try:
        # Load meal log
        if os.path.exists("user_meal_log.json"):
            with open("user_meal_log.json", "r") as f:
                meal_log = json.load(f)
        else:
            return {"message": "No meal data found"}
        
        # Use today's date if not specified
        if not date:
            from datetime import date as dt
            date = str(dt.today())
        
        # Get daily summary
        daily_summaries = meal_log.get("daily_summaries", {})
        if date not in daily_summaries:
            return {"message": f"No data found for {date}"}
        
        summary = daily_summaries[date]
        
        return ProgressResponse(
            date=date,
            daily_totals={
                "calories": summary.get("total_calories", 0),
                "protein": summary.get("total_protein", 0),
                "carbs": summary.get("total_carbs", 0),
                "fats": summary.get("total_fats", 0)
            },
            daily_percentages=summary.get("daily_percentages"),
            meal_breakdown=summary.get("meal_breakdown", {}),
            targets=summary.get("targets")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting progress: {str(e)}")

@app.get("/meals/history")
async def get_meal_history(days: int = 7, user_id: Optional[str] = "default"):
    """Get meal history for specified number of days"""
    try:
        # Load meal log
        if os.path.exists("user_meal_log.json"):
            with open("user_meal_log.json", "r") as f:
                meal_log = json.load(f)
        else:
            return {"message": "No meal data found"}
        
        # Get recent entries
        meal_entries = meal_log.get("meal_entries", {})
        daily_summaries = meal_log.get("daily_summaries", {})
        
        # Sort dates and get recent ones
        sorted_dates = sorted(meal_entries.keys(), reverse=True)[:days]
        
        history = []
        for date_str in sorted_dates:
            entries = meal_entries[date_str]
            summary = daily_summaries.get(date_str, {})
            
            history.append({
                "date": date_str,
                "meals": entries,
                "daily_totals": {
                    "calories": summary.get("total_calories", 0),
                    "protein": summary.get("total_protein", 0),
                    "carbs": summary.get("total_carbs", 0),
                    "fats": summary.get("total_fats", 0)
                },
                "meal_count": sum(len(meals) for meals in entries.values())
            })
        
        return {
            "history": history,
            "total_days": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting meal history: {str(e)}")

# Meal planning endpoints
@app.post("/meal-plan/generate")
async def generate_meal_plan(user_id: Optional[str] = "default"):
    """Generate a meal plan for the user"""
    if not PlanMyMeals_system:
        raise HTTPException(status_code=503, detail="PlanMyMeals system not initialized")
    
    try:
        # Request meal plan generation
        response = await PlanMyMeals_system.process_message_async("create meal plan")
        
        return {
            "message": "Meal plan generated",
            "meal_plan": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating meal plan: {str(e)}")

# Database search endpoint
@app.get("/foods/search")
async def search_foods(query: str, limit: int = 10):
    """Search for foods in the database"""
    try:
        import pandas as pd
        foods = []
        
        # Search calorie library
        if os.path.exists("calorie_library.csv"):
            df = pd.read_csv("calorie_library.csv")
            matches = df[df['Food Item'].str.contains(query, case=False, na=False)]
            for _, row in matches.head(limit//2).iterrows():
                foods.append({
                    "name": row.get('Food Item'),
                    "calories": float(row.get('Calories (Cal)', 0)),
                    "protein": float(row.get('Protein (g)', 0)),
                    "carbs": float(row.get('Carbs (g)', 0)),
                    "fats": float(row.get('Fats (g)', 0)),
                    "serving_size": row.get('Serving Size'),
                    "source": "calorie_library"
                })
        
        # Search Indian recipes
        if os.path.exists("indian_recipes.csv"):
            df = pd.read_csv("indian_recipes.csv")
            matches = df[df['Recipe Name'].str.contains(query, case=False, na=False)]
            for _, row in matches.head(limit//2).iterrows():
                foods.append({
                    "name": row.get('Recipe Name'),
                    "calories": float(row.get('Calories (per serving)', 0)),
                    "protein": float(row.get('Protein (g)', 0)),
                    "carbs": float(row.get('Carbs (g)', 0)),
                    "fats": float(row.get('Fats (g)', 0)),
                    "ingredients": row.get('Ingredients'),
                    "source": "indian_recipes"
                })
        
        return {
            "query": query,
            "results": foods[:limit],
            "total_found": len(foods)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching foods: {str(e)}")

# Reset/clear data endpoints
@app.delete("/reset/{user_id}")
async def reset_user_data(user_id: str, confirm: bool = False):
    """Reset all user data (use with caution)"""
    if not confirm:
        raise HTTPException(status_code=400, detail="Must set confirm=true to reset data")
    
    try:
        # Reset JSON files
        files_to_reset = ["user_info.json", "task_info.json", "user_meal_log.json"]
        
        for file_path in files_to_reset:
            if os.path.exists(file_path):
                if file_path == "user_meal_log.json":
                    default_data = {
                        "meal_entries": {},
                        "daily_summaries": {},
                        "last_updated": str(datetime.now().date())
                    }
                else:
                    default_data = {}
                
                with open(file_path, "w") as f:
                    json.dump(default_data, f, indent=2)
        
        return {"message": "User data reset successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting data: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "PlanMyMealsServer:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )