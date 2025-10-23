
import asyncio
import aiofiles
import json
import os
from typing import Dict, Any, List, Tuple, TypedDict
import openai
from datetime import datetime
from conversation_agent import PlanMyMealsState


class NutritionCalculator:
    """Core nutrition calculation functions"""
    
    @staticmethod
    def get_activity_multiplier(activity_level: str) -> float:
        """Get activity multiplier for TDEE calculation"""
        activity_levels = {
            "sedentary": 1.2,
            "lightly active": 1.375,
            "moderately active": 1.55,
            "very active": 1.725,
            "super active": 1.9
        }
        return activity_levels.get(activity_level.lower(), 1.2)
    
    @staticmethod
    def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
        if gender.lower() == 'male':
            return 10 * weight + 6.25 * height - 5 * age + 5
        elif gender.lower() == 'female':
            return 10 * weight + 6.25 * height - 5 * age - 161
        else:
            raise ValueError("Gender must be 'male' or 'female'.")
    
    @staticmethod
    def calculate_tdee(bmr: float, activity_multiplier: float) -> float:
        """Calculate Total Daily Energy Expenditure"""
        return bmr * activity_multiplier
    
    @staticmethod
    def suggest_calories(current_weight: float, goal_weight: float, tdee: float) -> float:
        """Suggest daily calories based on goals"""
        if current_weight > goal_weight:
            return tdee - 500  # Deficit for weight loss
        elif current_weight < goal_weight:
            return tdee + 500  # Surplus for weight gain
        else:
            return tdee  # Maintenance
    
    @staticmethod
    def distribute_calories(daily_calories: float) -> Dict[str, float]:
        """Distribute calories across meals"""
        return {
            "breakfast": daily_calories * 0.25,
            "lunch": daily_calories * 0.30,
            "dinner": daily_calories * 0.30,
            "snacks": daily_calories * 0.15
        }
    
    @staticmethod
    def calculate_macros(weight_kg: float, daily_calories: float, goal: str) -> Tuple[float, float, float]:
        """Calculate macronutrient distribution"""
        if goal == "lose":
            protein_per_kg = 2.0
            fat_percentage = 0.25
        elif goal == "maintain":
            protein_per_kg = 1.6
            fat_percentage = 0.25
        elif goal == "gain":
            protein_per_kg = 1.8
            fat_percentage = 0.22
        else:
            raise ValueError("Goal must be one of: lose, maintain, gain")
        
        protein_g = weight_kg * protein_per_kg
        protein_cal = protein_g * 4
        
        fat_cal = daily_calories * fat_percentage
        fat_g = fat_cal / 9
        
        carb_cal = daily_calories - (protein_cal + fat_cal)
        carb_g = carb_cal / 4
        
        return protein_g, carb_g, fat_g
    
    @staticmethod
    def macro_split_per_meal(protein: float, carbs: float, fats: float, 
                           meal_calories: Dict[str, float], total_calories: float) -> Dict[str, Dict[str, float]]:
        """Split macros per meal"""
        meal_macros = {}
        for meal, cal in meal_calories.items():
            ratio = cal / total_calories
            meal_macros[meal] = {
                "calories": cal,
                "protein": protein * ratio,
                "carbs": carbs * ratio,
                "fats": fats * ratio
            }
        return meal_macros


class ManagerAgent:
    def __init__(self, api_key: str = None):
        # Initialize OpenAI async client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.calculator = NutritionCalculator()
        
        # File paths
        self.user_info_file = "user_info.json"
        self.task_info_file = "task_info.json"
        self.meal_log_file = "user_meal_log.json"
        
        # Simple system prompt for decision making
        self.system_prompt = """You are a Manager Agent that decides what to do with user requests.

Available actions:
- meal_plan: User wants meal planning, recipes, nutrition advice
- meal_track: User wants to track meals, log food, monitor progress

Respond with just ONE word: "meal_plan" or "meal_track" or "unclear"

Examples:
- "create meal plan" → meal_plan
- "suggest recipes" → meal_plan  
- "track my breakfast" → meal_track
- "log my meals" → meal_track
- "I ate something" → meal_track
- "hello" → unclear"""
        
        # Initialize async
        asyncio.create_task(self.initialize_json_files_async())

    async def initialize_json_files_async(self):
        """Initialize JSON files asynchronously if they don't exist"""
        files_data = {
            self.user_info_file: {},
            self.task_info_file: {},
            self.meal_log_file: {}
        }
        
        for file_path, default_data in files_data.items():
            if not os.path.exists(file_path):
                try:
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(json.dumps(default_data, indent=2))
                    print(f"Manager: Created {file_path}")
                except Exception as e:
                    print(f"Manager: Error creating {file_path}: {str(e)}")

    async def load_json_async(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file asynchronously"""
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content) if content.strip() else {}
            return {}
        except Exception as e:
            print(f" Manager: Error loading {file_path}: {str(e)}")
            return {}

    async def save_json_async(self, file_path: str, data: Dict[str, Any]):
        """Save JSON file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            print(f" Manager: Saved {file_path}")
        except Exception as e:
            print(f" Manager: Error saving {file_path}: {str(e)}")

    async def load_user_info_async(self) -> Dict[str, Any]:
        """Load user_info.json asynchronously"""
        return await self.load_json_async(self.user_info_file)
    
    async def load_task_info_async(self) -> Dict[str, Any]:
        """Load task_info.json asynchronously"""
        return await self.load_json_async(self.task_info_file)
    
    async def save_user_info_async(self, data: Dict[str, Any]):
        """Save user_info.json asynchronously"""
        await self.save_json_async(self.user_info_file, data)
    
    async def save_task_info_async(self, data: Dict[str, Any]):
        """Save task_info.json asynchronously"""
        await self.save_json_async(self.task_info_file, data)
    
    async def check_profile_complete_async(self) -> Tuple[bool, List[str]]:
        """Check if user profile is complete"""
        user_info = await self.load_user_info_async()
        
        required_fields = [
            ("profile", "age"),
            ("profile", "gender"), 
            ("profile", "weight"),
            ("profile", "height"),
            ("profile", "activity_level"),
            ("goals", "goal_weight"),
            ("goals", "goal_type")
        ]
        
        missing = []
        for section, field in required_fields:
            if section not in user_info or user_info[section].get(field) is None:
                missing.append(f"{section}.{field}")
        
        return len(missing) == 0, missing
    
    async def check_calorie_bank_exists_async(self) -> bool:
        """Check if calorie bank and macros are calculated"""
        user_info = await self.load_user_info_async()
        return "calorie_bank" in user_info and "macros" in user_info
    
    async def calculate_nutrition_data_async(self) -> Dict[str, Any]:
        """Calculate BMR, TDEE, calories, and macros asynchronously"""
        user_info = await self.load_user_info_async()
        
        try:
            print("🔢 Manager: Calculating nutrition data...")
            
            # Get user data
            profile = user_info.get("profile", {})
            goals = user_info.get("goals", {})
            
            # Validate required data
            required_profile_fields = ["weight", "height", "age", "gender", "activity_level"]
            required_goals_fields = ["goal_weight", "goal_type"]
            
            for field in required_profile_fields:
                if field not in profile:
                    raise ValueError(f"Missing profile field: {field}")
            
            for field in required_goals_fields:
                if field not in goals:
                    raise ValueError(f"Missing goals field: {field}")
            
            # Calculate BMR
            bmr = self.calculator.calculate_bmr(
                profile["weight"],
                profile["height"], 
                profile["age"],
                profile["gender"]
            )
            print(f"Manager: BMR: {bmr:.1f}")
            
            # Calculate TDEE
            activity_multiplier = self.calculator.get_activity_multiplier(profile["activity_level"])
            tdee = self.calculator.calculate_tdee(bmr, activity_multiplier)
            print(f"Manager: TDEE: {tdee:.1f}")
            
            # Calculate daily calories
            daily_calories = self.calculator.suggest_calories(
                profile["weight"],
                goals["goal_weight"],
                tdee
            )
            print(f"Manager: Daily calories: {daily_calories:.1f}")
            
            # Distribute calories across meals
            meal_calories = self.calculator.distribute_calories(daily_calories)
            print(f"Manager: Meal distribution: {meal_calories}")
            
            # Calculate macros
            protein, carbs, fats = self.calculator.calculate_macros(
                profile["weight"],
                daily_calories,
                goals["goal_type"]
            )
            print(f"Manager: Macros - P:{protein:.1f}g C:{carbs:.1f}g F:{fats:.1f}g")
            
            # Split macros per meal
            meal_macros = self.calculator.macro_split_per_meal(
                protein, carbs, fats, meal_calories, daily_calories
            )
            
            # Add calculated data to user_info
            user_info["calorie_bank"] = meal_macros
            user_info["macros"] = {
                "daily_protein": protein,
                "daily_carbs": carbs,
                "daily_fats": fats
            }
            user_info["nutrition_targets"] = {
                "bmr": bmr,
                "tdee": tdee,
                "daily_calories": daily_calories,
                "calculated_at": datetime.now().isoformat()
            }
            
            # Save updated user_info
            await self.save_user_info_async(user_info)
            print("Manager: Saved nutrition data to user_info.json")
            
            return {
                "success": True,
                "bmr": bmr,
                "tdee": tdee,
                "daily_calories": daily_calories,
                "meal_macros": meal_macros
            }
            
        except Exception as e:
            print(f"Manager: Error calculating nutrition: {e}")
            return {"success": False, "error": str(e)}
    
    async def decide_action_async(self, user_input: str, task_type: str = None) -> str:
        """Use LLM to decide what action to take"""
        # If task_type is already determined, use it
        if task_type:
            return task_type
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            decision = response.choices[0].message.content.strip().lower()
            return decision
            
        except Exception as e:
            print(f"Manager: Error making decision: {e}")
            return "unclear"
    
    def create_agent_request(self, missing_fields: List[str]) -> Dict[str, Any]:
        """Create agent request for missing information"""
        return {
            "from_agent": "manager",
            "type": "user_info",
            "info_needed": missing_fields,
            "context": "completing user profile for task delegation"
        }
    
    async def process_request_async(self, user_input: str, task_type: str = None) -> Dict[str, Any]:
        """Main function to process requests and delegate with loop prevention"""
        print(f"Manager processing: '{user_input}' with task_type: {task_type}")
        
        try:
            # Step 1: Check if profile is complete
            profile_complete, missing_fields = await self.check_profile_complete_async()
            
            if not profile_complete:
                print(f"Manager: Profile incomplete. Missing: {missing_fields}")
                # Request missing info from conversationAgent
                agent_request = self.create_agent_request(missing_fields)
                return {
                    "status": "requesting_info",
                    "message": "I need some information to help you better. Let me gather the required details...",
                    "agent_request": agent_request,
                    "next_agent": "conversation",
                    "missing_fields": missing_fields
                }
            
            # Step 2: Decide what action to take
            decision = await self.decide_action_async(user_input, task_type)
            print(f"Manager: Decision: {decision}")
            
            if decision == "meal_plan":
                # Step 3: Check if nutrition data exists for meal planning
                if not await self.check_calorie_bank_exists_async():
                    print("Manager: Nutrition data missing. Calculating...")
                    calc_result = await self.calculate_nutrition_data_async()
                    
                    if not calc_result["success"]:
                        return {
                            "status": "calculation_error",
                            "message": f"I had trouble calculating your nutrition data: {calc_result['error']}. Let me help you step by step.",
                            "next_agent": "conversation",
                            "error": calc_result["error"]
                        }
                
                # Step 4: Delegate to meal_plan_agent
                return {
                    "status": "delegating",
                    "agent": "meal_plan_agent",
                    "action": "create_meal_plan",
                    "message": "Perfect! I have all your nutrition data ready. Creating your personalized meal plan now...",
                    "next_agent": "meal_plan",
                    "data_ready": True
                }
            
            elif decision == "meal_track":
                # Check if nutrition data exists for better tracking
                has_nutrition_data = await self.check_calorie_bank_exists_async()
                
                # Delegate to meal_track_agent
                return {
                    "status": "delegating", 
                    "agent": "meal_track_agent",
                    "action": "track_meals",
                    "message": "Great! I'll help you track your meals and monitor your nutrition progress...",
                    "next_agent": "meal_track",
                    "data_ready": has_nutrition_data
                }
            
            else:
                return {
                    "status": "unclear",
                    "message": "I'm not sure what you want to do. Try saying 'create meal plan' or 'track my meals' to get started! 🍽️",
                    "next_agent": "conversation"
                }
                
        except Exception as e:
            print(f"Manager: Error processing request: {e}")
            return {
                "status": "error",
                "message": "I encountered an error while processing your request. Let me redirect you to get help.",
                "next_agent": "conversation",
                "error": str(e)
            }

    async def handle_agent_request_async(self, state: PlanMyMealsState) -> PlanMyMealsState:
        """Handle agent requests to prevent loops"""
        agent_request = state.get("agent_request", {})
        
        if not agent_request:
            # No agent request, process normally
            return await self.process_delegation_async(state)
        
        # Check if this is a response to our previous request
        if agent_request.get("from_agent") == "manager":
            print("Manager: Received response to our request, processing normally...")
            # Clear the agent request and process normally
            state["agent_request"] = {}
            return await self.process_delegation_async(state)
        
        # This is a request from another agent
        print(f"Manager: Handling request from {agent_request.get('from_agent', 'unknown')}")
        
        # Process the request and delegate back
        return await self.process_delegation_async(state)

    async def process_delegation_async(self, state: PlanMyMealsState) -> PlanMyMealsState:
        """Main processing method for manager agent with loop prevention"""
        user_input = state["user_input"]
        task_type = state.get("task_type")
        
        print(f"Manager Agent processing: {user_input} (type: {task_type})")
        
        try:
            # Load current data
            user_info = await self.load_user_info_async()
            task_info = await self.load_task_info_async()
            
            # Update state with loaded data
            state["user_info"] = user_info
            state["task_info"] = task_info
            
            # Process the request
            result = await self.process_request_async(user_input, task_type)
            
            # Handle different result types
            if result["status"] == "requesting_info":
                # Need to request info from conversationAgent
                state["agent_request"] = result["agent_request"]
                state["current_agent"] = result["next_agent"]
                state["agent_response"] = result["message"]
                print(f"🔄 Manager: Requesting info, routing to {result['next_agent']}")
                
            elif result["status"] == "delegating":
                # Ready to delegate to specialized agent
                state["current_agent"] = result["next_agent"]
                state["agent_response"] = result["message"]
                
                # Update task_info with delegation details
                task_info["current_agent"] = result["agent"]
                task_info["action"] = result["action"]
                task_info["data_ready"] = result["data_ready"]
                task_info["delegated_at"] = datetime.now().isoformat()
                
                await self.save_task_info_async(task_info)
                state["task_info"] = task_info
                
                
                state["agent_request"] = {}
                
                print(f"🔄 Manager: Delegating to {result['next_agent']}")
                
            elif result["status"] == "calculation_error":
                # Error in calculations, go back to conversation
                state["current_agent"] = result["next_agent"]
                state["agent_response"] = result["message"]
                state["error_message"] = result["message"]
                
            
                state["agent_request"] = {}
                
                print(f"Manager: Calculation error, routing to {result['next_agent']}")
                
            else:
                # Unclear request or error, go back to conversation
                state["current_agent"] = result["next_agent"]
                state["agent_response"] = result["message"]
                
            
                state["agent_request"] = {}
                
                print(f"🔄 Manager: Unclear/error, routing to {result['next_agent']}")
            
            return state
            
        except Exception as e:
            error_msg = f"Manager Agent Error: {str(e)}"
            print(error_msg)
            state["error_message"] = error_msg
            state["agent_response"] = "I'm having trouble processing your request. Let me connect you with my colleague who can help."
            state["current_agent"] = "conversation"
            
            
            state["agent_request"] = {}
            
            return state


# Async node function for LangGraph
async def manager_node_async(state: PlanMyMealsState) -> PlanMyMealsState:
    """Async node function for manager agent in LangGraph with loop prevention"""
    print("Entering Manager Agent Node")
    
    # Get API key from environment
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        state["error_message"] = "OpenAI API key not found"
        state["agent_response"] = "I'm having configuration issues. Let me redirect you to get help."
        state["current_agent"] = "conversation"
    
        state["agent_request"] = {}
        return state
    
    # Create agent instance and process
    agent = ManagerAgent(api_key)
    
    # Handle agent requests to prevent loops
    result_state = await agent.handle_agent_request_async(state)
    
    print(f"Manager Agent completed - routing to: {result_state.get('current_agent', 'unknown')}")
    print(f"Manager Agent response: {result_state.get('agent_response', 'No response')[:100]}...")
    
    return result_state