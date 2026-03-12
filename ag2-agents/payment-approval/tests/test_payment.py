import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

def test_agents_instantiate():
    """Verify agent setup without initiating a chat."""
    import main as m
    assert m.researcher.name == "researcher"
    assert m.executor.name == "payment_executor"

def test_executor_human_input_mode():
    """Executor must have human_input_mode=ALWAYS — the approval gate."""
    import main as m
    assert m.executor.human_input_mode == "ALWAYS"

def test_researcher_termination_condition():
    import main as m
    assert m.researcher._is_termination_msg({"content": "Risk: low. ASSESSMENT COMPLETE"}) is True
    assert m.researcher._is_termination_msg({"content": "Still investigating..."}) is False

def test_executor_termination_condition():
    import main as m
    assert m.executor._is_termination_msg({"content": "Payment aborted. TERMINATE"}) is True
    assert m.executor._is_termination_msg({"content": "Please confirm."}) is False
