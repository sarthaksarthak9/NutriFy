import asyncio
import aiofiles
import json
import os
import openai
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, List, Tuple
from conversation_agent import PlanMyMealsState


class MealTrackAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        # File paths
        self.user_info_file = "user_info.json"
        self.meal_log_file = "user_meal_log.json"
        
        # Initialize datasets
        self.calorie_db = None
        self.indian_db = None

    async def load_json_async(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file asynchronously"""
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content) if content.strip() else {}
            return {}
        except Exception as e:
            print(f" Error loading {file_path}: {str(e)}")
            return {}

    async def save_json_async(self, file_path: str, data: Dict[str, Any]):
        """Save JSON file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            print(f" MealTrack saved {file_path}")
        except Exception as e:
            print(f" Error saving {file_path}: {str(e)}")

    async def initialize_meal_log_async(self):
        """Initialize meal log file"""
        if not os.path.exists(self.meal_log_file):
            default_log = {
                "meal_entries": {},
                "daily_summaries": {},
                "last_updated": str(date.today())
            }
            await self.save_json_async(self.meal_log_file, default_log)
            print("MealTrack: Created user_meal_log.json")

    async def load_datasets_async(self):
        """Load food databases"""
        loop = asyncio.get_event_loop()
        
        try:
            if os.path.exists("calorie_library.csv"):
                self.calorie_db = await loop.run_in_executor(None, pd.read_csv, "calorie_library.csv")
                print(f"MealTrack: Loaded calorie_library.csv: {len(self.calorie_db)} foods")
            else:
                self.calorie_db = pd.DataFrame()
            
            if os.path.exists("indian_recipes.csv"):
                self.indian_db = await loop.run_in_executor(None, pd.read_csv, "indian_recipes.csv")
                print(f"🍛 MealTrack: Loaded indian_recipes.csv: {len(self.indian_db)} recipes")
            else:
                self.indian_db = pd.DataFrame()
                
        except Exception as e:
            print(f"MealTrack: Error loading datasets: {e}")
            self.calorie_db = pd.DataFrame()
            self.indian_db = pd.DataFrame()

    async def get_openai_response_async(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Get LLM response"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"MealTrack OpenAI API Error: {str(e)}")
            return ""

    async def check_for_choice_response_async(self, user_input: str) -> bool:
        """Check if user input is a response to database choices"""
        choice_indicators = ["yes", "choose", "option", "edit", "none"]
        has_number = any(char.isdigit() for char in user_input)
        has_choice_word = any(word in user_input.lower() for word in choice_indicators)
        
        return has_choice_word or (has_number and len(user_input.strip()) <= 10)

    async def save_pending_matches_async(self, matches: Dict, meal_info: Dict) -> None:
        """Save pending matches to task_info.json for persistence"""
        task_info = await self.load_json_async("task_info.json")
        
        task_info["pending_matches"] = {
            "matches": matches,
            "meal_info": meal_info,
            "timestamp": str(datetime.now())
        }
        
        await self.save_json_async("task_info.json", task_info)
        print(" MealTrack: Saved pending matches to task_info.json")

    async def load_pending_matches_async(self) -> Dict[str, Any]:
        """Load pending matches from task_info.json"""
        task_info = await self.load_json_async("task_info.json")
        pending_data = task_info.get("pending_matches", {})
        
        if pending_data:
            print(" MealTrack: Loaded pending matches from task_info.json")
            return pending_data
        return {}

    async def clear_pending_matches_async(self) -> None:
        """Clear pending matches from task_info.json"""
        task_info = await self.load_json_async("task_info.json")
        
        if "pending_matches" in task_info:
            del task_info["pending_matches"]
            await self.save_json_async("task_info.json", task_info)
            print("🗑️ MealTrack: Cleared pending matches from task_info.json")

    async def create_matches_message_async(self, matches: Dict) -> str:
        """Create user-friendly message for database matches"""
        
        if not matches["found_matches"]:
            return " No matches found in database. Please provide nutrition info manually."
        
        message = f" Found {matches['total_matches']} matches:\n\n"
        
        for num, match in matches["matches"].items():
            message += f"**{num}. {match['name']}** ({match['source']})\n"
            message += f"    {match['calories']:.0f} cal | "
            message += f"Protein: {match['protein']:.1f}g | "
            message += f"Carbs: {match['carbs']:.1f}g | "
            message += f"Fats: {match['fats']:.1f}g\n"
            if match.get('serving_size'):
                message += f"   📏 Serving: {match['serving_size']}\n"
            message += "\n"
        
        message += " **Which one matches what you ate?**\n"
        message += "- Say 'yes option_no.' to choose option no.\n"
        message += "- Say 'edit no.: 300 calories' to choose option 2 with custom values\n"
        message += "- Say 'none' to enter manually\n"
        
        return message

    async def identify_meal_request_async(self, user_input: str) -> Dict[str, Any]:
        """Step 1: Identify if this is a meal tracking request"""
        
        prompt = f"""
        User said: "{user_input}"
        
        Determine if this is a meal tracking request and extract basic info.
        
        Return JSON:
        {{
            "is_meal_request": true or false,
            "food_name": "extracted food name" or null,
            "meal_type": "breakfast" or "lunch" or "dinner" or "snacks" or null,
            "servings": number or 1,
            "has_nutrition": true or false (if user provided calories/macros),
            "calories": number or null,
            "protein": number or null,
            "carbs": number or null,
            "fats": number or null
        }}
        
        Examples:
        - "meal track" → {{"is_meal_request": true, "food_name": null}}
        - "i had eggs" → {{"is_meal_request": true, "food_name": "eggs", "meal_type": "breakfast"}}
        - "track my lunch" → {{"is_meal_request": true, "food_name": null, "meal_type": "lunch"}}
        - "salami with 150 calories" → {{"is_meal_request": true, "food_name": "salami", "has_nutrition": true, "calories": 150}}
        """
        
        try:
            messages = [
                {"role": "system", "content": "Extract meal tracking info. Return only valid JSON without markdown."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.get_openai_response_async(messages)
            
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
            
        except Exception as e:
            print(f"Error identifying meal request: {e}")
            return {"is_meal_request": False}

    async def search_database_async(self, food_name: str) -> Dict[str, Any]:
        """Step 2: Search database and return matches in dict format"""
        
        if self.calorie_db is None:
            await self.load_datasets_async()
        
        matches = []
        
        # Search calorie library
        if not self.calorie_db.empty:
            for idx, row in self.calorie_db.iterrows():
                name = str(row.get('Food Item', '')).lower()
                if food_name.lower() in name or name in food_name.lower():
                    match = {
                        "id": f"cal_{idx}",
                        "name": row.get('Food Item', 'Unknown'),
                        "calories": float(row.get('Calories (Cal)', 0)),
                        "protein": float(row.get('Protein (g)', 0)),
                        "carbs": float(row.get('Carbs (g)', 0)),
                        "fats": float(row.get('Fats (g)', 0)),
                        "serving_size": row.get('Serving Size', 'N/A'),
                        "source": "calorie_library"
                    }
                    matches.append(match)
                    if len(matches) >= 5:  # Limit to 5 matches
                        break
        
        # Search Indian recipes if we need more matches
        if len(matches) < 5 and not self.indian_db.empty:
            for idx, row in self.indian_db.iterrows():
                name = str(row.get('Recipe Name', '')).lower()
                if food_name.lower() in name or name in food_name.lower():
                    match = {
                        "id": f"recipe_{idx}",
                        "name": row.get('Recipe Name', 'Unknown'),
                        "calories": float(row.get('Calories (per serving)', 0)),
                        "protein": float(row.get('Protein (g)', 0)),
                        "carbs": float(row.get('Carbs (g)', 0)),
                        "fats": float(row.get('Fats (g)', 0)),
                        "ingredients": row.get('Ingredients', ''),
                        "source": "indian_recipes"
                    }
                    matches.append(match)
                    if len(matches) >= 5:
                        break
        
        # Create database dict
        db_matches = {}
        for i, match in enumerate(matches, 1):
            db_matches[str(i)] = match
        
        return {
            "found_matches": len(matches) > 0,
            "total_matches": len(matches),
            "matches": db_matches
        }

    async def process_user_choice_async(self, user_input: str, stored_matches: Dict) -> Dict[str, Any]:
        """Step 3: Process user's choice (yes 1, yes 5, edit 2, etc.)"""
        
        prompt = f"""
        User said: "{user_input}"
        
        The user is responding to database food matches. Extract their choice.
        
        Return JSON:
        {{
            "action": "confirm" or "edit" or "manual" or "unknown",
            "choice_number": number or null,
            "edit_calories": number or null,
            "edit_protein": number or null,
            "edit_carbs": number or null,
            "edit_fats": number or null
        }}
        
        Examples:
        - "yes 1" → {{"action": "confirm", "choice_number": 1}}
        - "yes 5" → {{"action": "confirm", "choice_number": 5}}
        - "1" → {{"action": "confirm", "choice_number": 1}}
        - "edit 2: 300 calories" → {{"action": "edit", "choice_number": 2, "edit_calories": 300}}
        - "none" → {{"action": "manual"}}
        """
        
        try:
            messages = [
                {"role": "system", "content": "Extract user choice from database matches. Return only JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.get_openai_response_async(messages)
            
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            choice_data = json.loads(cleaned)
            
            # Validate choice number
            if choice_data.get("choice_number"):
                choice_num = str(choice_data["choice_number"])
                if choice_num in stored_matches:
                    return choice_data
            
            return choice_data
            
        except Exception as e:
            print(f"Error processing choice: {e}")
            return {"action": "unknown"}

    async def log_meal_async(self, meal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Log the meal to database"""
        
        await self.initialize_meal_log_async()
        meal_log = await self.load_json_async(self.meal_log_file)
        
        entry_date = str(date.today())
        meal_type = meal_data.get("meal_type", "snacks")
        
        # Initialize date structure
        if entry_date not in meal_log["meal_entries"]:
            meal_log["meal_entries"][entry_date] = {
                "breakfast": [], "lunch": [], "dinner": [], "snacks": []
            }
        
        # Create meal entry
        meal_entry = {
            "food_name": meal_data["food_name"],
            "servings": meal_data.get("servings", 1),
            "calories_per_serving": meal_data.get("calories", 0),
            "protein_per_serving": meal_data.get("protein", 0),
            "carbs_per_serving": meal_data.get("carbs", 0),
            "fats_per_serving": meal_data.get("fats", 0),
            "total_calories": meal_data.get("calories", 0) * meal_data.get("servings", 1),
            "total_protein": meal_data.get("protein", 0) * meal_data.get("servings", 1),
            "total_carbs": meal_data.get("carbs", 0) * meal_data.get("servings", 1),
            "total_fats": meal_data.get("fats", 0) * meal_data.get("servings", 1),
            "logged_at": str(datetime.now())
        }
        
        # Add to meal log
        meal_log["meal_entries"][entry_date][meal_type].append(meal_entry)
        
        # Save
        await self.save_json_async(self.meal_log_file, meal_log)
        
        return {
            "success": True,
            "message": f" Added {meal_data['food_name']} to {meal_type} for {entry_date}",
            "entry": meal_entry
        }

    async def process_meal_track_request_async(self, state: PlanMyMealsState) -> PlanMyMealsState:
        """Main processing method - simplified to avoid recursion"""
        user_input = state.get("user_input", "")
        print(f"📝 MealTrack Agent processing: {user_input}")

        try:
            # Step 1: Handle initial "meal track" request
            if user_input.lower().strip() in ["meal track", "track meals", "track"]:
                state["agent_response"] = "📝 Great! I'll help you track your meals. What did you eat? (e.g., 'I had eggs for breakfast')"
                state["current_agent"] = "meal_track"
                state["waiting_for_user_input"] = True
                return state
            
            # Step 2: Check for pending matches first (from task_info.json)
            pending_data = await self.load_pending_matches_async()
            
            if pending_data and "matches" in pending_data:
                print(f" MealTrack: Found pending matches, checking if this is a choice response")
                
                if await self.check_for_choice_response_async(user_input):
                    print(f" MealTrack: Processing choice response")
                    stored_matches = pending_data["matches"]
                    stored_meal_info = pending_data["meal_info"]
                    
                    choice_data = await self.process_user_choice_async(user_input, stored_matches["matches"])
                    print(f" MealTrack: Choice data: {choice_data}")
                    
                    if choice_data["action"] == "confirm":
                        # User chose a database match
                        choice_num = str(choice_data["choice_number"])
                        if choice_num in stored_matches["matches"]:
                            chosen_food = stored_matches["matches"][choice_num]
                            
                            # Prepare meal data
                            meal_data = {
                                "food_name": chosen_food["name"],
                                "meal_type": stored_meal_info.get("meal_type", "snacks"),
                                "servings": stored_meal_info.get("servings", 1),
                                "calories": chosen_food["calories"],
                                "protein": chosen_food["protein"],
                                "carbs": chosen_food["carbs"],
                                "fats": chosen_food["fats"]
                            }
                            
                            # Log the meal
                            result = await self.log_meal_async(meal_data)
                            
                            if result["success"]:
                                await self.clear_pending_matches_async()
                                state["agent_response"] = f"{result['message']}\n\n🎉 Meal logged successfully! What else can I help you with?"
                                state["current_agent"] = "conversation"
                                state["waiting_for_user_input"] = True
                                state["meal_logged"] = True
                                
                                print(f" MealTrack: Successfully logged meal choice {choice_num}")
                                return state
                        else:
                            state["agent_response"] = f" Invalid choice. Please choose between 1 and {len(stored_matches['matches'])}"
                            state["current_agent"] = "meal_track"
                            state["waiting_for_user_input"] = True
                            return state
                    
                    elif choice_data["action"] == "edit":
                        # User wants to edit a database match
                        choice_num = str(choice_data["choice_number"])
                        if choice_num in stored_matches["matches"]:
                            chosen_food = stored_matches["matches"][choice_num]
                            
                            # Use database values as base, override with user edits
                            meal_data = {
                                "food_name": chosen_food["name"],
                                "meal_type": stored_meal_info.get("meal_type", "snacks"),
                                "servings": stored_meal_info.get("servings", 1),
                                "calories": choice_data.get("edit_calories") or chosen_food["calories"],
                                "protein": choice_data.get("edit_protein") or chosen_food["protein"],
                                "carbs": choice_data.get("edit_carbs") or chosen_food["carbs"],
                                "fats": choice_data.get("edit_fats") or chosen_food["fats"]
                            }
                            
                            # Log the meal
                            result = await self.log_meal_async(meal_data)
                            
                            if result["success"]:
                                await self.clear_pending_matches_async()
                                edit_info = []
                                if choice_data.get("edit_calories"): edit_info.append(f"{choice_data['edit_calories']} cal")
                                if choice_data.get("edit_protein"): edit_info.append(f"{choice_data['edit_protein']}g protein")
                                if choice_data.get("edit_carbs"): edit_info.append(f"{choice_data['edit_carbs']}g carbs")
                                if choice_data.get("edit_fats"): edit_info.append(f"{choice_data['edit_fats']}g fats")
                                
                                edit_msg = f" with custom values: {', '.join(edit_info)}" if edit_info else ""
                                state["agent_response"] = f" Added {chosen_food['name']}{edit_msg} to {meal_data['meal_type']}\n\n🎉 Meal logged successfully! What else can I help you with?"
                                state["current_agent"] = "conversation"
                                state["waiting_for_user_input"] = True
                                state["meal_logged"] = True
                                
                                return state
                        
                    elif choice_data["action"] == "manual":
                        # User wants manual entry
                        await self.clear_pending_matches_async()
                        state["agent_request"] = {
                            "from_agent": "meal_track_agent",
                            "info_type": "manual_nutrition",
                            "food_name": stored_meal_info.get("food_name", "your food"),
                            "meal_type": stored_meal_info.get("meal_type", "meal"),
                            "context": "User chose manual entry. Please ask for nutrition info."
                        }
                        state["current_agent"] = "conversation"
                        state["waiting_for_user_input"] = False
                        return state
                    
                    else:
                        # Unknown action, show choices again
                        message = await self.create_matches_message_async(stored_matches)
                        state["agent_response"] = f" I didn't understand that.\n\n{message}"
                        state["current_agent"] = "meal_track"
                        state["waiting_for_user_input"] = True
                        return state
            
            # Step 3: Process new meal tracking request
            meal_request = await self.identify_meal_request_async(user_input)
            print(f" MealTrack: Meal request identified: {meal_request}")
            
            if not meal_request.get("is_meal_request"):
                state["agent_response"] = " I couldn't understand that. Try something like 'I had eggs for breakfast' or 'show my progress today'"
                state["current_agent"] = "meal_track"
                state["waiting_for_user_input"] = True
                return state
            
            # If user provided nutrition info directly, log it
            if meal_request.get("has_nutrition") and meal_request.get("food_name"):
                meal_data = {
                    "food_name": meal_request["food_name"],
                    "meal_type": meal_request.get("meal_type", "snacks"),
                    "servings": meal_request.get("servings", 1),
                    "calories": meal_request.get("calories", 0),
                    "protein": meal_request.get("protein", 0),
                    "carbs": meal_request.get("carbs", 0),
                    "fats": meal_request.get("fats", 0)
                }
                
                result = await self.log_meal_async(meal_data)
                
                if result["success"]:
                    state["agent_response"] = f"{result['message']}\n\n🎉 Meal logged successfully! What else can I help you with?"
                    state["current_agent"] = "conversation"
                    state["waiting_for_user_input"] = True
                    state["meal_logged"] = True
                    return state
            
            # If user mentioned food, search database
            if meal_request.get("food_name"):
                food_name = meal_request["food_name"]
                print(f" MealTrack: Searching database for: {food_name}")
                
                # Search database
                matches = await self.search_database_async(food_name)
                print(f" MealTrack: Found {matches['total_matches']} matches")
                
                if matches["found_matches"]:
                    # Save matches to task_info.json for persistence
                    await self.save_pending_matches_async(matches, meal_request)
                    
                    message = await self.create_matches_message_async(matches)
                    state["agent_response"] = message
                    state["current_agent"] = "meal_track"
                    state["waiting_for_user_input"] = True
                    
                    print(f" MealTrack: Saved matches to task_info and staying in meal_track")
                    return state
                else:
                    # No matches - ask for manual nutrition
                    state["agent_request"] = {
                        "from_agent": "meal_track_agent",
                        "info_type": "manual_nutrition",
                        "food_name": food_name,
                        "meal_type": meal_request.get("meal_type", "meal"),
                        "context": f"No database matches found for '{food_name}'. Need manual nutrition info."
                    }
                    state["current_agent"] = "conversation"
                    state["waiting_for_user_input"] = False
                    return state
            else:
                # No food mentioned, ask what they ate
                state["agent_response"] = "📝 What did you eat? Please tell me the food name (e.g., 'I had chicken curry for lunch')"
                state["current_agent"] = "meal_track"
                state["waiting_for_user_input"] = True
                return state

        except Exception as e:
            error_msg = f" MealTrack Agent Error: {str(e)}"
            print(error_msg)
            state["error_message"] = error_msg
            state["agent_response"] = "I'm having trouble tracking your meal. Please try again!"
            state["current_agent"] = "conversation"
            return state


# Async node function for LangGraph
async def meal_track_node_async(state: PlanMyMealsState) -> PlanMyMealsState:
    """Clean meal track node"""
    print("📝 Entering Meal Track Agent Node")
    
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        state["error_message"] = "OpenAI API key not found"
        state["agent_response"] = "I'm having configuration issues."
        state["current_agent"] = "conversation"
        return state
    
    agent = MealTrackAgent(api_key)
    result_state = await agent.process_meal_track_request_async(state)
    
    print(f" MealTrack Agent Response: {result_state.get('agent_response', 'No response')}")
    print(f" MealTrack Agent meal_logged: {result_state.get('meal_logged', False)}")
    
    return result_state