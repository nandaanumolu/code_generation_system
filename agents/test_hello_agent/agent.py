"""
Simple Test Agent to Verify ADK Setup
This agent just responds to greetings to test basic functionality
"""

from google.adk.agents import LlmAgent

# Create a simple test agent
root_agent = LlmAgent(
    name="test_hello_agent",
    model="gemini-2.0-flash",
    description="A simple test agent that greets users",
    instruction="""You are a friendly test agent. 
    When someone greets you, respond warmly and ask how you can help them today.
    Keep responses brief and friendly.
    
    Examples of greetings you might receive:
    - Hello
    - Hi
    - Hey
    - Good morning/afternoon/evening
    
    Always be polite and welcoming in your responses."""
)


# Test the agent setup
if __name__ == "__main__":
    print("Test Hello Agent is ready!")
    print("This agent responds to greetings.")
    print("\nTo run interactively:")
    print("  adk run agents/test_hello_agent")
    print("\nOr test with the web interface:")
    print("  adk web")