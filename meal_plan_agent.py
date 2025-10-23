import asyncio
import aiofiles
import json
import pandas as pd
import openai
import os
from typing import Dict, Any, List
from conversation_agent import PlanMyMealsState


class MealPlanAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        # File paths
        self.user_info_file = "user_info.json"
        self.task_info_file = "task_info.json"
        
        # Initialize datasets asynchronously
        self.calorie_db = None
        self.indian_db = None
        
        # System prompt for meal planning
        self.system_prompt = """You are a smart Meal Plan Agent that creates personalized meal suggestions.

🍽️ Your role:
- Analyze user's calorie targets and preferences
- Select 2 best recipe options for each meal from available datasets
- Consider nutritional balance and ingredient healthiness
- Explain why each option is healthy and suitable
- Handle requests for additional information from other agents

🧠 Selection criteria:
- Match calorie targets as closely as possible
- Respect dietary restrictions and preferences
- Choose nutritionally balanced options
- Consider ingredient quality and health benefits
- Provide variety across meals

🎯 Response format for each meal:
For each meal (breakfast, lunch, dinner, snacks), provide:
1. **Option 1**: Recipe name, calories, macros, why it's healthy
2. **Option 2**: Recipe name, calories, macros, why it's healthy

Be enthusiastic, use emojis, and explain the nutritional benefits of each choice!

🔄 Agent Communication:
- If you need additional user information, request it from conversationAgent
- Provide clear context about what information you need and why"""

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
            print(f" MealPlan saved {file_path}")
        except Exception as e:
            print(f" Error saving {file_path}: {str(e)}")

    async def load_datasets_async(self):
        """Load CSV datasets asynchronously using thread pool"""
        loop = asyncio.get_event_loop()
        
        try:
            # Load calorie library
            if os.path.exists("calorie_library.csv"):
                self.calorie_db = await loop.run_in_executor(None, pd.read_csv, "calorie_library.csv")
                print(f" MealPlan: Loaded calorie_library.csv: {len(self.calorie_db)} recipes")
            else:
                self.calorie_db = pd.DataFrame()
                print(" MealPlan: calorie_library.csv not found")
            
            # Load Indian recipes
            if os.path.exists("indian_recipes.csv"):
                self.indian_db = await loop.run_in_executor(None, pd.read_csv, "indian_recipes.csv")
                print(f" MealPlan: Loaded indian_recipes.csv: {len(self.indian_db)} recipes")
            else:
                self.indian_db = pd.DataFrame()
                print(" MealPlan: indian_recipes.csv not found")
                
        except Exception as e:
            print(f" MealPlan: Error loading datasets: {e}")
            self.calorie_db = pd.DataFrame()
            self.indian_db = pd.DataFrame()

    async def get_openai_response_async(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """Get response from OpenAI API asynchronously"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" MealPlan OpenAI API Error: {str(e)}")
            return f"I'm having trouble generating your meal plan. Error: {str(e)}"

    async def find_semantic_matches_async(self, food_name: str, all_foods: List[Dict]) -> List[Dict]:
        """Use LLM to find semantically similar foods asynchronously"""
        if not all_foods:
            return []
        
        # Create food list for LLM
        food_list = [f"{i+1}. {food['name']}" for i, food in enumerate(all_foods)]
        food_list_text = "\n".join(food_list[:50])  # Limit to first 50 foods
        
        matching_prompt = f"""
        Looking for foods similar to: "{food_name}"
        
        Available foods in database:
        {food_list_text}
        
        Find the TOP 5 most semantically similar foods from the list above.
        Consider:
        - Similar ingredients (lemon mojito ≈ lemon soda)
        - Similar cooking methods (fried ≈ pan-fried)
        - Similar food types (chicken curry ≈ chicken masala)
        - Similar drinks (coffee ≈ espresso)
        
        Return ONLY a JSON array with the numbers of matching foods, ranked by similarity:
        [1, 15, 23, 7, 12]
        
        If no good matches found, return: []
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert at finding semantically similar foods. Return only a JSON array of numbers."},
                {"role": "user", "content": matching_prompt}
            ]
            
            response = await self.get_openai_response_async(messages, temperature=0.2, max_tokens=100)
            match_indices = json.loads(response)
            
            # Convert to actual food objects
            matches = []
            for idx in match_indices:
                if 1 <= idx <= len(all_foods):
                    food = all_foods[idx - 1].copy()
                    food["match_rank"] = len(matches) + 1  # Add ranking
                    matches.append(food)
            
            return matches[:5]  # Return top 5
            
        except Exception as e:
            print(f" Error finding semantic matches: {e}")
            
            # Fallback: simple string matching
            matches = []
            food_lower = food_name.lower()
            for food in all_foods[:20]:  # Check first 20 foods
                if food_lower in food["name"].lower() or food["name"].lower() in food_lower:
                    matches.append(food)
            
            return matches[:5]

    async def get_recipes_for_meal_async(self, meal_type: str, target_calories: float, target_protein: float, 
                           target_carbs: float, target_fats: float, preferences: Dict) -> List[Dict]:
        """Get recipe options for a specific meal asynchronously"""
        # Ensure datasets are loaded
        if self.calorie_db is None or self.indian_db is None:
            await self.load_datasets_async()
        
        all_recipes = []
        
        # Process calorie database
        if not self.calorie_db.empty:
            for _, row in self.calorie_db.iterrows():
                recipe = {
                    "name": row.get('Food Item', 'Unknown'),
                    "calories": float(row.get('Calories (Cal)', 0)),
                    "protein": float(row.get('Protein (g)', 0)),
                    "carbs": float(row.get('Carbs (g)', 0)),
                    "fats": float(row.get('Fats (g)', 0)),
                    "serving_size": row.get('Serving Size', 'N/A'),
                    "ingredients": row.get('Ingredients', ''),
                    "source": "calorie_library"
                }
                all_recipes.append(recipe)
        
        # Process Indian recipes database
        if not self.indian_db.empty:
            for _, row in self.indian_db.iterrows():
                recipe = {
                    "name": row.get('Recipe Name', 'Unknown'),
                    "calories": float(row.get('Calories (per serving)', 0)),
                    "protein": float(row.get('Protein (g)', 0)),
                    "carbs": float(row.get('Carbs (g)', 0)),
                    "fats": float(row.get('Fats (g)', 0)),
                    "ingredients": row.get('Ingredients', ''),
                    "procedure": row.get('Procedure', ''),
                    "source": "indian_recipes"
                }
                all_recipes.append(recipe)
        
        # Filter based on preferences
        filtered_recipes = self.filter_by_preferences(all_recipes, preferences)
        
        # Calculate match scores and sort
        scored_recipes = []
        for recipe in filtered_recipes:
            score = self.calculate_match_score(recipe, target_calories, target_protein, target_carbs, target_fats)
            recipe["match_score"] = score
            scored_recipes.append(recipe)
        
        # Sort by best match (lowest score)
        scored_recipes.sort(key=lambda x: x["match_score"])
        
        return scored_recipes[:10]  # Return top 10 matches

    def filter_by_preferences(self, recipes: List[Dict], preferences: Dict) -> List[Dict]:
        """Filter recipes based on dietary restrictions and preferences"""
        filtered = []
        
        dietary_restrictions = preferences.get("dietary_restrictions", [])
        allergies = preferences.get("allergies", [])
        
        for recipe in recipes:
            ingredients = recipe.get("ingredients", "").lower()
            
            # Check allergies
            if any(allergy.lower() in ingredients for allergy in allergies):
                continue
            
            # Check dietary restrictions
            skip_recipe = False
            for restriction in dietary_restrictions:
                if restriction.lower() == "vegetarian" and any(meat in ingredients for meat in ["chicken", "beef", "pork", "fish", "meat"]):
                    skip_recipe = True
                    break
                elif restriction.lower() == "vegan" and any(animal in ingredients for animal in ["milk", "cheese", "egg", "butter", "meat", "chicken", "fish"]):
                    skip_recipe = True
                    break
            
            if not skip_recipe:
                filtered.append(recipe)
        
        return filtered

    def calculate_match_score(self, recipe: Dict, target_cal: float, target_protein: float, 
                            target_carbs: float, target_fats: float) -> float:
        """Calculate how well a recipe matches targets (lower is better)"""
        cal_diff = abs(recipe["calories"] - target_cal) / max(target_cal, 1)
        protein_diff = abs(recipe["protein"] - target_protein) / max(target_protein, 1)
        carbs_diff = abs(recipe["carbs"] - target_carbs) / max(target_carbs, 1)
        fats_diff = abs(recipe["fats"] - target_fats) / max(target_fats, 1)
        
        # Weight calories more heavily
        return (cal_diff * 0.5) + (protein_diff * 0.2) + (carbs_diff * 0.2) + (fats_diff * 0.1)

    async def create_meal_plan_with_llm_async(self, meal_options: Dict[str, List[Dict]], user_preferences: Dict) -> str:
        """Use LLM to create a well-formatted meal plan with health insights asynchronously"""
        
        # Prepare meal options data for LLM
        meal_data = {}
        for meal_type, recipes in meal_options.items():
            meal_data[meal_type] = {
                "target_calories": recipes[0].get("target_calories", 0) if recipes else 0,
                "options": recipes[:2]  # Top 2 options
            }
        
        context = f"""
        User preferences: {json.dumps(user_preferences, indent=2)}
        
        Available meal options: {json.dumps(meal_data, indent=2)}
        
        Create a personalized meal plan with 2 options for each meal. For each option:
        1. Explain why it's a healthy choice
        2. Highlight key nutritional benefits
        3. Mention how it fits their goals
        4. Use emojis and be enthusiastic!
        """
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]
            
            return await self.get_openai_response_async(messages, temperature=0.7, max_tokens=1500)
            
        except Exception as e:
            print(f" Error generating meal plan: {e}")
            return self.create_simple_meal_plan(meal_options)

    def create_simple_meal_plan(self, meal_options: Dict[str, List[Dict]]) -> str:
        """Fallback: Create simple meal plan without LLM"""
        meal_plan = "YOUR PERSONALIZED MEAL PLAN\n"
        meal_plan += "=" * 50 + "\n\n"
        
        for meal_type, recipes in meal_options.items():
            meal_plan += f"🍴 **{meal_type.upper()}:**\n"
            
            for i, recipe in enumerate(recipes[:2], 1):
                meal_plan += f"   {i}. {recipe['name']}\n"
                meal_plan += f"       {recipe['calories']:.0f} cal | "
                meal_plan += f"Protein: {recipe['protein']:.1f}g | "
                meal_plan += f"Carbs: {recipe['carbs']:.1f}g | "
                meal_plan += f"Fats: {recipe['fats']:.1f}g\n"
                if recipe.get('ingredients'):
                    meal_plan += f"       Ingredients: {recipe['ingredients'][:100]}...\n"
                meal_plan += "\n"
        
        return meal_plan

    def request_missing_info(self, missing_fields: List[str], context: str) -> Dict[str, Any]:
        """Request missing information from conversationAgent"""
        return {
            "from_agent": "meal_plan_agent",
            "info_type": "specific_fields",
            "fields": missing_fields,
            "context": context
        }

    async def check_required_data_async(self) -> tuple[bool, List[str], str]:
        """Check if all required data is available for meal planning asynchronously"""
        user_info = await self.load_json_async(self.user_info_file)
        
        # Check for calorie bank
        if "calorie_bank" not in user_info:
            return False, ["calorie_bank"], "calorie and macro targets"
        
        # Check for preferences
        if "preferences" not in user_info:
            return False, ["preferences"], "dietary preferences and restrictions"
        
        return True, [], ""

    async def create_meal_plan_async(self, user_input: str = "") -> Dict[str, Any]:
        """Main function to create meal plan asynchronously"""
        print("MealPlanAgent: Creating personalized meal plan...")
        
        # Check if all required data is available
        data_complete, missing_fields, context = await self.check_required_data_async()
        
        if not data_complete:
            print(f" MealPlanAgent: Missing data: {missing_fields}")
            # Request missing info from conversationAgent
            agent_request = self.request_missing_info(missing_fields, f"meal planning - need {context}")
            return {
                "success": False,
                "message": f"I need {context} to create your meal plan. Let me get that information...",
                "agent_request": agent_request,
                "next_agent": "conversation"
            }
        
        # Load user data
        user_info = await self.load_json_async(self.user_info_file)
        calorie_bank = user_info["calorie_bank"]
        preferences = user_info.get("preferences", {})
        
        print(f"MealPlanAgent: User preferences: {preferences}")
        
        # Get recipe options for each meal
        meal_options = {}
        
        for meal_type, meal_data in calorie_bank.items():
            print(f"🔍 MealPlanAgent: Finding recipes for {meal_type}...")
            
            recipes = await self.get_recipes_for_meal_async(
                meal_type,
                meal_data["calories"],
                meal_data["protein"], 
                meal_data["carbs"],
                meal_data["fats"],
                preferences
            )
            
            # Add target info to recipes
            for recipe in recipes:
                recipe["target_calories"] = meal_data["calories"]
            
            meal_options[meal_type] = recipes
            print(f"MealPlanAgent: Found {len(recipes)} options for {meal_type}")
        
        # Create meal plan using LLM
        meal_plan_text = await self.create_meal_plan_with_llm_async(meal_options, preferences)
        
        return {
            "success": True,
            "message": " Your personalized meal plan is ready!",
            "meal_plan": meal_plan_text,
            "meal_options": meal_options,
            "next_agent": "conversation",
            "conversation_complete": True
        }

    async def handle_follow_up_async(self, user_input: str) -> Dict[str, Any]:
        """Handle follow-up questions about the meal plan asynchronously"""
        #handle requests like "change breakfast options" or "vegetarian alternatives"
        
        if any(word in user_input.lower() for word in ["change", "different", "alternative", "other"]):
            return {
                "success": True,
                "message": "I can help you customize your meal plan! What would you like to change? ",
                "next_agent": "meal_plan"
            }
        elif any(word in user_input.lower() for word in ["recipe", "ingredients", "how to make"]):
            return {
                "success": True,
                "message": "I'd be happy to share recipe details! Which recipe would you like to know more about? 📝",
                "next_agent": "meal_plan"
            }
        else:
            return {
                "success": True,
                "message": "Your meal plan is complete! You can ask me to modify it or start tracking your meals. What would you like to do next? 😊",
                "next_agent": "conversation"
            }

    async def process_meal_plan_request_async(self, state: PlanMyMealsState) -> PlanMyMealsState:
        """Main processing method for meal plan agent"""
        user_input = state.get("user_input", "")
        print(f"MealPlan Agent processing: {user_input}")

        try:
            # Check if this is an initial meal plan request or follow-up
            if state.get("agent_request") and state["agent_request"].get("from_agent") == "meal_plan_agent":
                # This is a response to our request for missing info
                result = await self.create_meal_plan_async(user_input)
            else:
                # Check if meal plan already exists - handle follow-up
                user_info = await self.load_json_async(self.user_info_file)
                if "calorie_bank" in user_info and any(word in user_input.lower() for word in ["change", "different", "recipe", "how"]):
                    result = await self.handle_follow_up_async(user_input)
                else:
                    # Create new meal plan
                    result = await self.create_meal_plan_async(user_input)
            
            # Update state based on result
            if result["success"]:
                state["agent_response"] = result["message"]
                if "meal_plan" in result:
                    state["agent_response"] += f"\n\n{result['meal_plan']}"
                
                state["current_agent"] = result.get("next_agent", "conversation")
                state["conversation_complete"] = result.get("conversation_complete", False)
                
            else:
                # Need to request info from conversationAgent
                if "agent_request" in result:
                    state["agent_request"] = result["agent_request"]
                    state["current_agent"] = result["next_agent"]
                    state["agent_response"] = result["message"]
                else:
                    state["agent_response"] = result["message"]
                    state["current_agent"] = result.get("next_agent", "conversation")
            
            # Update user_info in state
            state["user_info"] = await self.load_json_async(self.user_info_file)
            
            return state

        except Exception as e:
            error_msg = f" MealPlan Agent Error: {str(e)}"
            print(error_msg)
            state["error_message"] = error_msg
            state["agent_response"] = "I'm having trouble creating your meal plan. Please try again!"
            state["current_agent"] = "conversation"
            return state


# Async node function for LangGraph
async def meal_plan_node_async(state: PlanMyMealsState) -> PlanMyMealsState:
    """Async LangGraph node for MealPlanAgent"""
    print(" Entering Meal Plan Agent Node")
    
    # Get API key from environment
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        state["error_message"] = "OpenAI API key not found"
        state["agent_response"] = "I'm having configuration issues. Please check the API key setup."
        return state
    
    # Create agent instance and process
    agent = MealPlanAgent(api_key)
    result_state = await agent.process_meal_plan_request_async(state)
    
    print(f"MealPlan Agent Response: {result_state.get('agent_response', 'No response')}")
    return result_state