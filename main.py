import os
import asyncio
import aiofiles
import json
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import all agents (these will need to be async too)
from conversation_agent import PlanMyMealsState, conversation_node_async, ConversationAgent
from manager_agent import manager_node_async, ManagerAgent  
from meal_plan_agent import meal_plan_node_async, MealPlanAgent
from meal_track_agent import meal_track_node_async, MealTrackAgent
from dotenv import load_dotenv
import openai

class PlanMyMealsSystem:
    def __init__(self, api_key: str = None):
        load_dotenv()  # Load .env variables
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Use async OpenAI client
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize individual agents for setup
        self.personality_agent = ConversationAgent(self.api_key)
        self.manager_agent = ManagerAgent(self.api_key)
        self.meal_plan_agent = MealPlanAgent(self.api_key)
        self.meal_track_agent = MealTrackAgent(self.api_key)
        
        # Build the LangGraph workflow
        self.workflow = self.build_workflow()
        
        print("PlanMyMeals System initialized successfully!")
        print("All agents loaded and ready!")
        print("LangGraph workflow built!")

    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all agents and routing"""
        
        # Create the state graph
        workflow = StateGraph(PlanMyMealsState)
        
        # Add all agent nodes 
        workflow.add_node("conversation", conversation_node_async)
        workflow.add_node("manager", manager_node_async)
        workflow.add_node("meal_plan", meal_plan_node_async)
        workflow.add_node("meal_track", meal_track_node_async)
        
        # Define routing logic
        def route_from_conversation(state: PlanMyMealsState) -> str:
            """Route from conversation agent based on current state"""
            current_agent = state.get("current_agent", "conversation")
            
            print(f"Routing from conversation to: {current_agent}")
            
            # If we're waiting for user input, END the workflow
            if state.get("waiting_for_user_input", False):
                print("Waiting for user input - ending workflow")
                return END
            
            # If agent request exists, stay in conversation to handle it
            if state.get("agent_request"):
                return "conversation"
            
            # Route based on current_agent field
            if current_agent == "manager":
                return "manager"
            elif current_agent == "meal_plan":
                return "meal_plan" 
            elif current_agent == "meal_track":
                return "meal_track"
            else:
                return END
        
        def route_from_manager(state: PlanMyMealsState) -> str:
            """Route from manager agent based on delegation decision"""
            current_agent = state.get("current_agent", "conversation")
            
            print(f"Routing from manager to: {current_agent}")
            
            # If agent request exists, route to conversation
            if state.get("agent_request"):
                return "conversation"
            
            # Route based on current_agent field
            if current_agent == "meal_plan":
                return "meal_plan"
            elif current_agent == "meal_track":
                return "meal_track"
            else:
                return "conversation"
        
        def route_from_meal_plan(state: PlanMyMealsState) -> str:
            """Route from meal plan agent"""
            current_agent = state.get("current_agent", "conversation")
            
            print(f"Routing from meal_plan to: {current_agent}")
            
            # If agent request exists, route to conversation
            if state.get("agent_request"):
                return "conversation"
            
            # If conversation is complete, end
            if state.get("conversation_complete"):
                return END
            
            # Otherwise route based on current_agent
            if current_agent == "meal_plan":
                return "meal_plan"
            else:
                return "conversation"
        
        def route_from_meal_track(state: PlanMyMealsState) -> str:
            """Route from meal track agent"""
            current_agent = state.get("current_agent", "conversation")
            
            print(f"Routing from meal_track to: {current_agent}")
            
            # If we're waiting for user input, END the workflow
            if state.get("waiting_for_user_input", False):
                print("MealTrack waiting for user input - ending workflow")
                return END
            
            # If agent request exists, route to conversation
            if state.get("agent_request"):
                return "conversation"
            
            # Route based on current_agent field
            if current_agent == "meal_track":
                return "meal_track"
            else:
                return "conversation"
        
        # Add conditional edges with routing logic
        workflow.add_conditional_edges(
            "conversation",
            route_from_conversation,
            {
                "conversation": "conversation",
                "manager": "manager", 
                "meal_plan": "meal_plan",
                "meal_track": "meal_track",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "meal_plan",
            route_from_meal_plan,
            {
                "conversation": "conversation",
                "meal_plan": "meal_plan",
                END: END
            }
        )
        
        workflow.add_conditional_edges(
            "meal_track", 
            route_from_meal_track,
            {
                "conversation": "conversation",
                "meal_track": "meal_track",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "manager",
            route_from_manager,
            {
                "conversation": "conversation",
                "meal_plan": "meal_plan",
                "meal_track": "meal_track",
                END: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("conversation")
        
        return workflow

    def create_initial_state(self, user_input: str) -> PlanMyMealsState:
        """Create initial state for the conversation"""
        return {
            "user_input": user_input,
            "current_agent": "conversation",
            "task_type": None,
            "user_info": {},
            "task_info": {},
            "meal_log": {},
            "agent_response": "",
            "info_needed": [],
            "agent_request": {},
            "conversation_complete": False,
            "error_message": "",
            "current_matches": None,
            "current_meal_info": None,
            "waiting_for_user_input": False,
            "meal_logged": False,  # NEW
            "last_processed_input": "",  # NEW
            "stored_matches": None,  # NEW: For clean meal track agent
            "stored_meal_info": None  # NEW: For clean meal track agent
        }

    async def process_message_async(self, user_input: str, config: Dict[str, Any] = None) -> str:
        """Process a message asynchronously through the workflow"""
        print(f"\nUser: {user_input}")
        print("=" * 50)
        
        # Create initial state
        initial_state = self.create_initial_state(user_input)
        
        # Configure workflow execution
        if config is None:
            config = {"recursion_limit": 10}
        
        try:
            # Compile workflow with memory checkpointing
            memory = MemorySaver()
            app = self.workflow.compile(checkpointer=memory)
            
            # Execute workflow
            thread_id = "PlanMyMeals_conversation"
            final_state = None
            
            async for state in app.astream(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            ):
                print(f"State update: {list(state.keys())}")
                final_state = state
            
            # Extract response from final state
            if final_state:
                # Get the last state values
                last_state_key = list(final_state.keys())[-1]
                response_state = final_state[last_state_key]
                
                response = response_state.get("agent_response", "I'm processing your request...")
                
                print(f"PlanMyMeals: {response}")
                return response
            else:
                return "I'm having trouble processing your request. Please try again."
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(error_msg)
            return "Sorry, I encountered an error. Please try again."

    def process_message(self, user_input: str) -> str:
        """Synchronous wrapper for processing messages"""
        return asyncio.run(self.process_message_async(user_input))

    async def run_interactive_session_async(self):
        """Run an interactive conversation session asynchronously"""
        print("\n" + "="*60)
        print("Welcome to PlanMyMeals - Your AI Nutrition Assistant!")
        print("="*60)
        print("\nI can help you with:")
        print("   • Creating personalized meal plans")
        print("   • Tracking your daily meals and nutrition")
        print("   • Calculating your calorie and macro targets")
        print("   • Finding recipes from food databases")
        print("\nTry saying:")
        print("   • 'Hello' or 'meal plan' to get started")
        print("   • 'I had eggs for breakfast' to track meals")
        print("   • 'help' for more options")
        print("   • 'quit' to exit")
        print("\n" + "="*60)
        
        conversation_count = 0
        
        while True:
            try:
                # Get user input (this remains sync as it's console input)
                user_input = input(f"\n[{conversation_count + 1}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle system commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nThank you for using PlanMyMeals!")
                    print("Keep up the great work with your nutrition goals!")
                    print("Stay healthy and see you next time!")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'status':
                    await self.show_system_status_async()
                    continue
                
                elif user_input.lower() == 'reset':
                    await self.reset_conversation_async()
                    continue
                
                # Process the message through LangGraph workflow
                response = await self.process_message_async(user_input)
                
                conversation_count += 1
                
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Take care of your health!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                print("Please try again or type 'help' for assistance.")

    def run_interactive_session(self):
        """Synchronous wrapper for interactive session"""
        return asyncio.run(self.run_interactive_session_async())

    def show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("PlanMyMeals HELP")
        print("="*50)
        print("\n**Agent Commands:**")
        print("   • 'meal plan' - Create a personalized meal plan")
        print("   • 'track meals' - Start tracking your daily food intake")
        print("   • 'I had [food] for [meal]' - Log what you ate")
        print("   • 'show progress' - View your daily nutrition progress")
        print("\n**System Commands:**")
        print("   • 'help' - Show this help message")
        print("   • 'status' - Show current system status")
        print("   • 'reset' - Reset all conversation data")
        print("   • 'quit' - Exit PlanMyMeals")
        print("\n**Example Conversations:**")
        print("   • 'I'm 25, male, 70kg, want to lose weight'")
        print("   • 'I had scrambled eggs for breakfast'")
        print("   • 'Create a meal plan for me'")
        print("   • 'Show my progress today'")
        print("="*50)

    async def show_system_status_async(self):
        """Show current system status asynchronously"""
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        
        # Check JSON files asynchronously
        files_status = {}
        required_files = ["user_info.json", "task_info.json", "user_meal_log.json"]
        
        for file in required_files:
            if os.path.exists(file):
                try:
                    async with aiofiles.open(file, 'r') as f:
                        content = await f.read()
                        data = json.loads(content)
                        files_status[file] = f"Loaded ({len(str(data))} chars)"
                except Exception as e:
                    files_status[file] = f"Error reading: {str(e)}"
            else:
                files_status[file] = "Not found"
        
        print("\n**JSON Files:**")
        for file, status in files_status.items():
            print(f"   {file}: {status}")
        
        # Check CSV databases
        csv_files = ["calorie_library.csv", "indian_recipes.csv"]
        print("\n**Database Files:**")
        for file in csv_files:
            if os.path.exists(file):
                try:
                    import pandas as pd
                    loop = asyncio.get_event_loop()
                    df = await loop.run_in_executor(None, pd.read_csv, file)
                    print(f"   {file}: Loaded ({len(df)} rows)")
                except Exception as e:
                    print(f"   {file}: Error reading: {str(e)}")
            else:
                print(f"   {file}: Not found")
        
        # Check API key
        print(f"\n**API Key:** {'Configured' if self.api_key else 'Missing'}")
        
        print("="*50)

    async def reset_conversation_async(self):
        """Reset conversation state asynchronously"""
        confirm = input("\n⚠️  Are you sure you want to reset all conversation data? (yes/no): ")
        
        if confirm.lower() in ['yes', 'y']:
            try:
                # Reset JSON files by reinitializing them
                await self.conversation_agent.initialize_json_files_async()
                print("Conversation data reset successfully!")
                print("You can start fresh now.")
            except Exception as e:
                print(f"Error resetting data: {str(e)}")
        else:
            print("Reset cancelled.")

async def main_async():
    """Main async entry point for PlanMyMeals"""
    try:
        # Initialize the system
        print("Starting PlanMyMeals...")

        # Check for API key
        load_dotenv()  # Load .env variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not found!")
            print("Please set your API key:")
            print("   1. Create a .env file with: OPENAI_API_KEY=your-api-key-here")
            print("   2. Or export OPENAI_API_KEY='your-api-key-here'")
            print("   3. Or set it in your environment variables")
            return 1
        
        # Create and run the system
        PlanMyMeals = PlanMyMealsSystem(api_key)
        await PlanMyMeals.run_interactive_session_async()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! Stay healthy!")
        return 0
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please check your setup and try again.")
        return 1

def main():
    """Main entry point for PlanMyMeals"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    exit(main())