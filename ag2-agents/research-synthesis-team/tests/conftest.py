"""
Ensure the project root (research-synthesis-team/) is on sys.path so that
local modules (agents.py, agent_executor.py, workflow.py) are importable.
Also evict the installed 'agents' package (OpenAI Agents SDK) that would
otherwise shadow the local agents.py.
"""
import sys
import os

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent not in sys.path:
    sys.path.insert(0, parent)

# Evict the installed 'agents' package from sys.modules cache so
# 'from agents import build_agents' finds our local agents.py instead.
for key in list(sys.modules.keys()):
    if key == "agents" or key.startswith("agents."):
        mod = sys.modules[key]
        if hasattr(mod, "__file__") and mod.__file__ and parent not in (mod.__file__ or ""):
            del sys.modules[key]
