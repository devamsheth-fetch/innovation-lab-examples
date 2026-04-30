import logging

# Ensure logging is set up if needed
logging.basicConfig(level=logging.INFO)

from src.agents.architect import architect_agent

if __name__ == "__main__":
    import uvicorn
    # The uagents Agent.run() method handles the server startup.
    # We just import the pre-configured architect_agent and run it.
    architect_agent.run()
