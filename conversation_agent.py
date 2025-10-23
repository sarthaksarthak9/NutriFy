import asyncio
import aiofiles
import json
import os
from typing import Dict, Any, TypedDict, List
import openai
from datetime import datetime


class PlanMyMealsState(TypedDict):
    """State definition for the PlanMyMeals system"""
    user_input: str
    current_agent: str
    task_type: str
    user_info: Dict[str, Any]
    task_info: Dict[str, Any]
    meal_log: Dict[str, Any]
    agent_response: str
    info_needed: List[str]
    agent_request: Dict[str, Any]
    conversation_complete: bool
    error_message: str
    current_matches: Any
    current_meal_info: Any
    waiting_for_user_input: bool
    meal_logged: bool  
    last_processed_input: str  


class ConversationAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # File paths
        self.user_info_file = "user_info.json"
        self.task_info_file = "task_info.json"
        self.meal_log_file = "user_meal_log.json"
        
        # Initialize async
        asyncio.create_task(self.initialize_json_files_async())

    async def initialize_json_files_async(self):
        """Initialize JSON files asynchronously if they don't exist"""
        files_data = {
            self.user_info_file: {
                "profile": {
                    "age": None,
                    "gender": None,
                    "weight": None,
                    "height": None,
                    "activity_level": None
                },
                "goals": {
                    "goal_weight": None,
                    "goal_type": None
                },
                "preferences": {
                    "dietary_restrictions": [],
                    "allergies": []
                }
            },
            self.task_info_file: {
                "profile_complete": False,
                "wants_meal_plan": None,
                "missing_fields": [],
                "current_step": "greeting",
                "current_agent": "conversation",
                "task_type": None
            },
            self.meal_log_file: {
                "meal_entries": {},
                "daily_summaries": {},
                "last_updated": str(datetime.now().date())
            }
        }
        
        for file_path, default_data in files_data.items():
            if not os.path.exists(file_path):
                try:
                    async with aiofiles.open(file_path, 'w') as f:
                        await f.write(json.dumps(default_data, indent=2))
                    print(f" conversation: Created {file_path}")
                except Exception as e:
                    print(f" conversation: Error creating {file_path}: {str(e)}")

    async def load_json_async(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file asynchronously"""
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content) if content.strip() else {}
            return {}
        except Exception as e:
            print(f" conversation: Error loading {file_path}: {str(e)}")
            return {}

    async def save_json_async(self, file_path: str, data: Dict[str, Any]):
        """Save JSON file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            print(f"💾 conversation: Saved {file_path}")
        except Exception as e:
            print(f" conversation: Error saving {file_path}: {str(e)}")

    async def get_openai_response_async(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Get response from OpenAI API asynchronously"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f" conversation: OpenAI API Error: {str(e)}")
            return f"I'm having trouble connecting to my AI service. Error: {str(e)}"

    async def check_missing_fields_async(self, user_info: Dict[str, Any]) -> List[str]:
        """Check which required fields are still missing"""
        missing = []
        
        # Check profile fields
        required_profile = ["age", "gender", "weight", "height", "activity_level"]
        profile = user_info.get("profile", {})
        for field in required_profile:
            if not profile.get(field):
                missing.append(f"profile.{field}")
        
        # Check goal fields
        required_goals = ["goal_weight", "goal_type"]
        goals = user_info.get("goals", {})
        for field in required_goals:
            if not goals.get(field):
                missing.append(f"goals.{field}")
        
        return missing

    async def extract_user_info_async(self, user_input: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to extract and update user information from natural language"""
        missing_fields = await self.check_missing_fields_async(user_info)
        
        system_prompt = f"""Extract user information from their input and return ONLY a JSON object.

Current missing fields: {missing_fields}
Current user info: {json.dumps(user_info, indent=2)}

Rules for extraction:
- Only include fields you can confidently extract
- For age: integer between 10-100
- For weight/goal_weight: number between 30-200 (in kg)
- For height: number between 120-220 (in cm)
- For gender: "male" or "female"
- For activity_level: "sedentary", "lightly active", "moderately active", "very active", or "super active"
- For goal_type: "lose", "maintain", or "gain"

Example output:
{{
    "profile": {{
        "age": 25,
        "weight": 70
    }},
    "goals": {{
        "goal_type": "lose"
    }}
}}

If no information can be extracted, return: {{"extracted": false}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User input: {user_input}"}
        ]

        try:
            response = await self.get_openai_response_async(messages, temperature=0.1)
            print(f" conversation: LLM extraction response: {response}")
            extracted_data = json.loads(response)
            
            if extracted_data.get("extracted") == False:
                print(" conversation: No information extracted")
                return {"updated": False, "fields": []}
            
            # Update user_info with extracted data
            updated_fields = []
            updated = False
            
            # Ensure user_info has proper structure
            if "profile" not in user_info:
                user_info["profile"] = {}
            if "goals" not in user_info:
                user_info["goals"] = {}
            if "preferences" not in user_info:
                user_info["preferences"] = {}
            
            if "profile" in extracted_data:
                print(f" conversation: Updating profile with: {extracted_data['profile']}")
                for key, value in extracted_data["profile"].items():
                    if value is not None:
                        print(f" conversation: Setting profile.{key} = {value}")
                        user_info["profile"][key] = value
                        updated_fields.append(f"profile.{key}")
                        updated = True
            
            if "goals" in extracted_data:
                print(f" conversation: Updating goals with: {extracted_data['goals']}")
                for key, value in extracted_data["goals"].items():
                    if value is not None:
                        print(f" conversation: Setting goals.{key} = {value}")
                        user_info["goals"][key] = value
                        updated_fields.append(f"goals.{key}")
                        updated = True
            
            if "preferences" in extracted_data:
                print(f" conversation: Updating preferences with: {extracted_data['preferences']}")
                for key, value in extracted_data["preferences"].items():
                    if isinstance(value, list):
                        if key not in user_info["preferences"]:
                            user_info["preferences"][key] = []
                        user_info["preferences"][key].extend(value)
                        updated_fields.append(f"preferences.{key}")
                        updated = True
            
            print(f" conversation: Updated fields: {updated_fields}")
            print(f" conversation: Final user_info: {json.dumps(user_info, indent=2)}")
            
            return {"updated": updated, "fields": updated_fields, "user_info": user_info}
            
        except Exception as e:
            print(f"conversation: Error extracting info: {str(e)}")
            return {"updated": False, "fields": []}

    async def generate_conversation_response_async(self, user_input: str, user_info: Dict, missing_fields: List[str], context: str = "") -> str:
        """Generate response using LLM"""
        
        system_prompt = """You are PlanMyMeals, a friendly and knowledgeable nutrition assistant. You have a warm, encouraging conversation and use emojis appropriately.

Your core traits:
- Encouraging and supportive 🌟
- Knowledgeable about nutrition 🍎
- Asks clarifying questions when needed ❓
- Celebrates user progress 🎉
- Uses food and health emojis naturally 🥗

Your main job is to collect user profile information before they can access meal planning or tracking features.

Keep responses conversational and helpful. If you need to collect user information, ask in a friendly way."""

        user_context = f"""
Current user info: {json.dumps(user_info, indent=2)}
Missing profile fields: {missing_fields}
Profile complete: {len(missing_fields) == 0}
Additional context: {context}
User Input: "{user_input}"

Provide a helpful, friendly response as PlanMyMeals."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context}
        ]

        return await self.get_openai_response_async(messages, temperature=0.7)

    async def determine_next_agent_async(self, user_input: str, user_info: Dict[str, Any], missing_fields: List[str]) -> str:
        """Determine which agent should handle the request next"""
        
        # CRITICAL: If profile is not complete, ALWAYS stay with conversation
        if missing_fields:
            print(f"🔒 conversation: Profile incomplete, staying in conversation. Missing: {missing_fields}")
            return "conversation"
        
        # Profile is complete, check for task requests
        user_input_lower = user_input.lower()
        
        # Check for meal planning requests
        meal_plan_keywords = ["meal plan", "create meal", "diet plan", "plan meals", "nutrition plan"]
        if any(keyword in user_input_lower for keyword in meal_plan_keywords):
            print("🍽️ conversation: Meal plan requested, routing to manager")
            return "manager"
        
        # Check for meal tracking requests
        meal_track_keywords = ["track", "log", "ate", "had", "food", "breakfast", "lunch", "dinner", "snack", "progress"]
        if any(keyword in user_input_lower for keyword in meal_track_keywords):
            print(" conversation: Meal tracking requested, routing to manager")
            return "manager"
        
        # Default: stay with conversation for more conversation
        print(" conversation: General conversation, staying in conversation")
        return "conversation"

    async def handle_agent_request_async(self, agent_request: Dict[str, Any], user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle information requests from other agents"""
        from_agent = agent_request.get("from_agent", "unknown")
        request_type = agent_request.get("type", "unknown")
        info_type = agent_request.get("info_type", "unknown")  # Also check info_type
        info_needed = agent_request.get("info_needed", [])
        
        print(f" conversation: Handling request from {from_agent}, type: {request_type}, info_type: {info_type}")
        
        if request_type == "user_info" or info_type == "missing_user_data":
            missing_fields = await self.check_missing_fields_async(user_info)
            
            if missing_fields:
                # Still missing required info
                friendly_fields = []
                for field in missing_fields:
                    if "profile.age" in field:
                        friendly_fields.append("age")
                    elif "profile.gender" in field:
                        friendly_fields.append("gender")
                    elif "profile.weight" in field:
                        friendly_fields.append("weight (in kg)")
                    elif "profile.height" in field:
                        friendly_fields.append("height (in cm)")
                    elif "profile.activity_level" in field:
                        friendly_fields.append("activity level")
                    elif "goals.goal_weight" in field:
                        friendly_fields.append("goal weight")
                    elif "goals.goal_type" in field:
                        friendly_fields.append("goal (lose, maintain, or gain weight)")
                
                message = f"I need some more information to help you better. Could you please tell me your {', '.join(friendly_fields)}? 📝"
                
                return {
                    "status": "need_info",
                    "message": message,
                    "missing_fields": missing_fields
                }
            else:
                # All info is complete
                return {
                    "status": "info_complete",
                    "message": "Perfect! I have all the information needed. ",
                    "user_info": user_info
                }
        
        elif info_type == "meal_details":
            # MealTrackAgent wants user to describe their meal
            return {
                "status": "need_info",
                "message": "📝 What did you eat? Please describe your meal like 'I had eggs for breakfast' or 'I ate chicken curry for dinner' 🍽️"
            }
        
        elif info_type == "manual_nutrition":
            # MealTrackAgent needs manual nutrition info
            food_name = agent_request.get("food_name", "your food")
            meal_type = agent_request.get("meal_type", "")
            
            message = f" I couldn't find '{food_name}' in my database. Please provide the nutritional information manually:\n\n"
            message += f"For example:\n"
            message += f"• 'I had {food_name} with 200 calories, 20g protein, 30g carbs, 10g fats'\n"
            message += f"• 'I had {food_name} with 150 calories'\n"
            message += f"• '{food_name} has 180 calories and 25g protein'\n\n"
            message += f"Just tell me the nutrition info and I'll log it for you! "
            
            return {
                "status": "need_info",
                "message": message
            }
        
        elif info_type == "specific_fields":
            # Check if specific fields are available
            fields = agent_request.get("fields", [])
            context = agent_request.get("context", "")
            missing_requested = []
            
            for field in fields:
                if "." in field:  # Handle nested fields like "profile.weight"
                    section, key = field.split(".")
                    if user_info.get(section, {}).get(key) is None:
                        missing_requested.append(field)
                else:
                    if user_info.get(field) is None:
                        missing_requested.append(field)
            
            if missing_requested:
                question = f"I need to know your {', '.join(missing_requested)} to help with {context}. Could you provide this information? 🤔"
                return {
                    "status": "need_info",
                    "message": question,
                    "missing_fields": missing_requested
                }
            else:
                return {
                    "status": "info_complete",
                    "message": "I have all the information needed! ",
                    "user_info": user_info
                }
        
        # Handle other request types
        return {
            "status": "unknown_request",
            "message": f"I received a request from {from_agent} but I'm not sure how to handle it. Let me help you directly! "
        }

    async def process_user_input_async(self, state: PlanMyMealsState) -> PlanMyMealsState:
        """Main processing method for conversation agent"""
        user_input = state["user_input"]
        print(f" conversation Agent processing: {user_input}")

        try:
            # Load current data
            user_info = await self.load_json_async(self.user_info_file)
            task_info = await self.load_json_async(self.task_info_file)
            
            # Update state with loaded data
            state["user_info"] = user_info
            state["task_info"] = task_info

            # CHECK FOR REVERT_MEAL_TRACK FLAG FIRST
            if task_info.get("revert_meal_track") == True:
                print(f" conversation: revert_meal_track=True, routing directly to meal_track with input: {user_input}")
                state["current_agent"] = "meal_track"
                state["waiting_for_user_input"] = False
                # Don't change user_input, pass it as-is to meal_track
                return state

            # Handle agent requests FIRST 
            if state.get("agent_request"):
                agent_request = state["agent_request"]
                print(f" conversation: Processing agent request: {agent_request}")
                
                request_result = await self.handle_agent_request_async(agent_request, user_info)
                
                if request_result["status"] == "need_info":
                    # Need more user info
                    state["agent_response"] = request_result["message"]
                    state["current_agent"] = "conversation"
                    state["waiting_for_user_input"] = True
                    #
                    return state
                elif request_result["status"] == "info_complete":
                    # Info is complete, send back to requesting agent
                    state["agent_response"] = request_result["message"] 
                    state["current_agent"] = agent_request.get("from_agent", "manager")
                    state["waiting_for_user_input"] = False
                    
                    state["agent_request"] = {}
                    return state
                else:
                    # Unknown request, clear and continue normally
                    state["agent_response"] = request_result["message"]
                    state["current_agent"] = "conversation"
                    state["waiting_for_user_input"] = True
                    
                    state["agent_request"] = {}
                    return state

            # CHECK FOR DUPLICATE MEAL LOGGING
            last_processed_input = state.get("last_processed_input", "")
            meal_logged = state.get("meal_logged", False)
            
            # If this is the same input and meal was already logged, don't process again
            if user_input == last_processed_input and meal_logged:
                print(f" conversation: Duplicate input detected and meal already logged, staying in conversation")
                response = "Great! Your meal has been logged successfully. What else can I help you with? "
                state["agent_response"] = response
                state["current_agent"] = "conversation"
                state["waiting_for_user_input"] = True
                # Clear the meal_logged flag for future inputs
                state["meal_logged"] = False
                state["last_processed_input"] = ""
                return state

            # Normal user interaction processing
            
            # Step 1: Try to extract information from user input
            extraction_result = await self.extract_user_info_async(user_input, user_info)
            
            if extraction_result.get("updated"):
                user_info = extraction_result["user_info"]
                await self.save_json_async(self.user_info_file, user_info)
                state["user_info"] = user_info
                print(f"converation: Updated fields: {extraction_result['fields']}")
            
            # Step 2: Check what's still missing
            missing_fields = await self.check_missing_fields_async(user_info)
            
            # Step 3: Update task info
            task_info["missing_fields"] = missing_fields
            task_info["profile_complete"] = len(missing_fields) == 0
            
            # Step 4: Determine next agent based on profile completeness
            next_agent = await self.determine_next_agent_async(user_input, user_info, missing_fields)
            
            # Step 5: Generate appropriate response
            if next_agent == "manager" and len(missing_fields) == 0:
                # Profile complete, can delegate to manager
                # Set task type based on user input
                user_input_lower = user_input.lower()
                if any(keyword in user_input_lower for keyword in ["meal plan", "create meal", "diet plan"]):
                    task_info["task_type"] = "meal_plan"
                    state["task_type"] = "meal_plan"
                elif any(keyword in user_input_lower for keyword in ["track", "log", "ate", "had"]):
                    task_info["task_type"] = "meal_track"
                    state["task_type"] = "meal_track"
                
                # STORE INPUT FOR DUPLICATE DETECTION
                state["last_processed_input"] = user_input
                state["meal_logged"] = False  # Will be set to True by meal track agent
                
                response = " Perfect! I have all your information. Let me connect you with the right specialist..."
                state["current_agent"] = "manager"
                state["waiting_for_user_input"] = False
            else:
                # Stay in conversation
                context = f"Missing fields: {missing_fields}" if missing_fields else "Profile complete"
                response = await self.generate_conversation_response_async(user_input, user_info, missing_fields, context)
                state["current_agent"] = "conversation"
                state["waiting_for_user_input"] = True
            
            # Step 6: Update state and save
            state["agent_response"] = response
            task_info["current_agent"] = state["current_agent"]
            await self.save_json_async(self.task_info_file, task_info)
            state["task_info"] = task_info
            
            # ALWAYS clear agent_request to prevent loops
            state["agent_request"] = {}
            
            return state

        except Exception as e:
            error_msg = f"conversation Agent Error: {str(e)}"
            print(error_msg)
            state["error_message"] = error_msg
            state["agent_response"] = "I'm having some technical difficulties. Please try again!"
            state["current_agent"] = "conversation"
            state["waiting_for_user_input"] = True
            # CLEAR agent_request on error
            state["agent_request"] = {}
            return state


# Async node function for LangGraph
async def conversation_node_async(state: PlanMyMealsState) -> PlanMyMealsState:
    """Async node function for conversation agent in LangGraph"""
    print("🤖 Entering conversation Agent Node")
    
    # Get API key from environment
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        state["error_message"] = "OpenAI API key not found"
        state["agent_response"] = "I'm having configuration issues. Please check the API key setup."
        state["current_agent"] = "conversation"
        state["waiting_for_user_input"] = True
        # CLEAR agent_request on error
        state["agent_request"] = {}
        return state
    
    # Create agent instance and process
    agent = ConversationAgent(api_key)
    result_state = await agent.process_user_input_async(state)
    
    print(f" conversation Agent Response: {result_state.get('agent_response', 'No response')}")
    print(f" conversation Agent Next: {result_state.get('current_agent', 'Unknown')}")
    
    return result_state