"""
Ensure payment-approval/ is on sys.path so local modules are importable.
"""
import sys
import os

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent not in sys.path:
    sys.path.insert(0, parent)

# Evict the installed 'agents' package (OpenAI Agents SDK) from sys.modules
# so 'from agents import build_agents' finds our local agents.py instead.
for key in list(sys.modules.keys()):
    if key == "agents" or key.startswith("agents."):
        mod = sys.modules[key]
        if hasattr(mod, "__file__") and mod.__file__ and parent not in (mod.__file__ or ""):
            del sys.modules[key]
