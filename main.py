import asyncio
import os
from typing import Any, Dict, Optional

import openai
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app_json_async import DEFAULT_USER_ID, init_app_storage_sync, reset_app_documents_sync
from conversation_agent import ConversationAgent, PlanMyMealsState, conversation_node_async
from db import crud
from db.session import SessionLocal
from manager_agent import ManagerAgent, manager_node_async
from meal_plan_agent import MealPlanAgent, meal_plan_node_async
from meal_track_agent import MealTrackAgent, meal_track_node_async


class PlanMyMealsSystem:
    def __init__(self, api_key: str = None):
        load_dotenv()
        init_app_storage_sync()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY in your environment or .env file."
            )

        self.client = openai.AsyncOpenAI(api_key=self.api_key)

        self.conversation_agent = ConversationAgent(self.api_key)
        self.manager_agent = ManagerAgent(self.api_key)
        self.meal_plan_agent = MealPlanAgent(self.api_key)
        self.meal_track_agent = MealTrackAgent(self.api_key)

        self.workflow = self._build_workflow()

        print("PlanMyMeals system initialised — all agents ready!")

    # ── Workflow ──────────────────────────────────────────────────────────────

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(PlanMyMealsState)

        workflow.add_node("conversation", conversation_node_async)
        workflow.add_node("manager", manager_node_async)
        workflow.add_node("meal_plan", meal_plan_node_async)
        workflow.add_node("meal_track", meal_track_node_async)

        workflow.add_conditional_edges("conversation", self._route_from_conversation, {
            "conversation": "conversation",
            "manager": "manager",
            "meal_plan": "meal_plan",
            "meal_track": "meal_track",
            END: END,
        })
        workflow.add_conditional_edges("manager", self._route_from_manager, {
            "conversation": "conversation",
            "meal_plan": "meal_plan",
            "meal_track": "meal_track",
            END: END,
        })
        workflow.add_conditional_edges("meal_plan", self._route_from_meal_plan, {
            "conversation": "conversation",
            "meal_plan": "meal_plan",
            END: END,
        })
        workflow.add_conditional_edges("meal_track", self._route_from_meal_track, {
            "conversation": "conversation",
            "meal_track": "meal_track",
            END: END,
        })

        workflow.set_entry_point("conversation")
        return workflow

    @staticmethod
    def _route_from_conversation(state: PlanMyMealsState) -> str:
        current = state.get("current_agent", "conversation")
        print(f"Routing from conversation → {current}")

        if state.get("waiting_for_user_input"):
            return END
        if state.get("agent_request"):
            return "conversation"
        return current if current in ("manager", "meal_plan", "meal_track") else END

    @staticmethod
    def _route_from_manager(state: PlanMyMealsState) -> str:
        current = state.get("current_agent", "conversation")
        print(f"Routing from manager → {current}")

        if state.get("agent_request"):
            return "conversation"
        return current if current in ("meal_plan", "meal_track") else "conversation"

    @staticmethod
    def _route_from_meal_plan(state: PlanMyMealsState) -> str:
        current = state.get("current_agent", "conversation")
        print(f"Routing from meal_plan → {current}")

        if state.get("agent_request"):
            return "conversation"
        if state.get("conversation_complete"):
            return END
        return "meal_plan" if current == "meal_plan" else "conversation"

    @staticmethod
    def _route_from_meal_track(state: PlanMyMealsState) -> str:
        current = state.get("current_agent", "conversation")
        print(f"Routing from meal_track → {current}")

        if state.get("waiting_for_user_input"):
            return END
        if state.get("agent_request"):
            return "conversation"
        return "meal_track" if current == "meal_track" else "conversation"

    # ── State ─────────────────────────────────────────────────────────────────

    def _initial_state(self, user_input: str) -> PlanMyMealsState:
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
            "meal_logged": False,
            "last_processed_input": "",
            "stored_matches": None,
            "stored_meal_info": None,
        }

    # ── Message processing ────────────────────────────────────────────────────

    async def process_message_async(
        self, user_input: str, config: Optional[Dict[str, Any]] = None
    ) -> str:
        print(f"\nUser: {user_input}\n{'=' * 50}")

        initial_state = self._initial_state(user_input)
        run_config = config or {"recursion_limit": 10}

        try:
            memory = MemorySaver()
            compiled = self.workflow.compile(checkpointer=memory)
            final_state = None

            async for state in compiled.astream(
                initial_state,
                config={"configurable": {"thread_id": "nutrify_session"}, **run_config},
            ):
                print(f"State update: {list(state.keys())}")
                final_state = state

            if final_state:
                last_key = list(final_state.keys())[-1]
                response = final_state[last_key].get("agent_response", "Still processing — please wait.")
                print(f"PlanMyMeals: {response}")
                return response

            return "I couldn't process that request. Please try again."

        except Exception as e:
            print(f"Workflow error: {e}")
            return "Something went wrong. Please try again."

    def process_message(self, user_input: str) -> str:
        return asyncio.run(self.process_message_async(user_input))

    # ── Interactive CLI session ───────────────────────────────────────────────

    async def run_interactive_session_async(self):
        print("\n" + "=" * 60)
        print("  Welcome to PlanMyMeals — Your AI Nutrition Assistant!")
        print("=" * 60)
        print("\nI can help you with:")
        print("  • Creating personalised meal plans")
        print("  • Tracking your daily meals and nutrition")
        print("  • Calculating calorie and macro targets")
        print("  • Finding recipes from food databases")
        print("\nTry:")
        print("  'Hello' or 'meal plan' to get started")
        print("  'I had eggs for breakfast' to track a meal")
        print("  'help' for all commands  •  'quit' to exit")
        print("=" * 60)

        count = 0
        while True:
            try:
                user_input = input(f"\n[{count + 1}] You: ").strip()
                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd in ("quit", "exit", "bye", "goodbye"):
                    print("\nThanks for using PlanMyMeals — stay healthy!")
                    break
                elif cmd == "help":
                    self._show_help()
                    continue
                elif cmd == "status":
                    await self._show_status_async()
                    continue
                elif cmd == "reset":
                    await self._reset_async()
                    continue

                await self.process_message_async(user_input)
                count += 1
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\nGoodbye! Take care of your health!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Please try again or type 'help' for assistance.")

    def run_interactive_session(self):
        asyncio.run(self.run_interactive_session_async())

    def _show_help(self):
        print("\n" + "=" * 50)
        print("  PlanMyMeals — Help")
        print("=" * 50)
        print("\nAgent commands:")
        print("  meal plan        → Create a personalised meal plan")
        print("  track meals      → Start tracking daily food intake")
        print("  I had X for Y    → Log a meal (e.g. 'eggs for breakfast')")
        print("  show progress    → View today's nutrition summary")
        print("\nSystem commands:")
        print("  help   → Show this message")
        print("  status → Show system status")
        print("  reset  → Wipe all conversation data")
        print("  quit   → Exit")
        print("\nExample phrases:")
        print("  'I'm 25, male, 70 kg, want to lose weight'")
        print("  'I had scrambled eggs for breakfast'")
        print("  'Create a meal plan for me'")
        print("=" * 50)

    async def _show_status_async(self):
        print("\n" + "=" * 50)
        print("  System Status")
        print("=" * 50)

        db = SessionLocal()
        try:
            doc_map = {
                "user_info": "user_info.json",
                "task_info": "task_info.json",
                "meal_log":  "user_meal_log.json",
            }
            print("\nApp documents (database):")
            for doc_key, label in doc_map.items():
                data = crud.get_document(db, DEFAULT_USER_ID, doc_key)
                status = f"loaded ({len(str(data))} chars)" if data is not None else "not found"
                print(f"  {label}: {status}")
        except Exception as e:
            print(f"  Error reading database: {e}")
        finally:
            db.close()

        print("\nCSV databases:")
        for csv in ("calorie_library.csv", "indian_recipes.csv"):
            if os.path.exists(csv):
                try:
                    import pandas as pd
                    df = await asyncio.get_running_loop().run_in_executor(None, pd.read_csv, csv)
                    print(f"  {csv}: loaded ({len(df)} rows)")
                except Exception as e:
                    print(f"  {csv}: error — {e}")
            else:
                print(f"  {csv}: not found")

        print(f"\nAPI key: {'configured' if self.api_key else 'MISSING'}")
        print("=" * 50)

    async def _reset_async(self):
        confirm = input("\nAre you sure you want to reset all data? (yes/no): ")
        if confirm.strip().lower() in ("yes", "y"):
            reset_app_documents_sync()
            print("Data reset. You can start fresh now.")
        else:
            print("Reset cancelled.")


# ── Entry points ──────────────────────────────────────────────────────────────

async def main_async() -> int:
    print("Starting PlanMyMeals...")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "OpenAI API key not found.\n"
            "Add it to a .env file as: OPENAI_API_KEY=your-key-here"
        )
        return 1

    try:
        system = PlanMyMealsSystem(api_key)
        await system.run_interactive_session_async()
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye! Stay healthy!")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
