"""
Simple Test Agent to Verify ADK Setup
This agent just responds to greetings to test basic functionality
"""

from google.adk import Agent

# Create a simple test agent
root_agent = Agent(
    name="test_hello_agent",
    model="gemini-2.0-flash",
    description="A simple test agent that greets users",
    instruction="""You are a friendly test agent. 
    When someone greets you, respond warmly and ask how you can help them today.
    Keep responses brief and friendly."""
)