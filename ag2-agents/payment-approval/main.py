"""
AG2 two-agent payment approval with human-in-the-loop gate.

researcher — investigates the recipient and produces a risk assessment
executor   — presents the assessment and waits for explicit human confirmation
             before executing the Skyfire payment

human_input_mode="ALWAYS" on executor is the approval gate: the agent
pauses before every response so the human can type "yes" to proceed or
"no" to abort. No custom routing logic required.
"""
import os
from dotenv import load_dotenv
from autogen import ConversableAgent, LLMConfig

load_dotenv()

llm_config = LLMConfig(
    {
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    },
    temperature=0.2,
    cache_seed=None,
)

researcher = ConversableAgent(
    name="researcher",
    system_message=(
        "You are a payment risk analyst. Investigate the payment recipient using available "
        "tools: check their Fetch.ai address history, reputation, and any known flags. "
        "Produce a concise risk assessment with a clear recommendation (proceed / do not proceed). "
        "End your assessment with ASSESSMENT COMPLETE."
    ),
    llm_config=llm_config,
    # Conversation ends via max_turns or executor's TERMINATE check.
)

executor = ConversableAgent(
    name="payment_executor",
    system_message=(
        "You handle payment execution. Present the researcher's risk assessment clearly, "
        "state the exact payment details (recipient, amount, reason), then ask the human "
        "to confirm. If the human approves, call the skyfire_send tool to execute the payment. "
        "If the human declines, acknowledge and terminate. End with TERMINATE."
    ),
    llm_config=llm_config,
    human_input_mode="ALWAYS",  # pauses before every response — the human types yes/no
    is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
)


def run_payment_approval(recipient: str, amount: float, reason: str) -> None:
    researcher.initiate_chat(
        executor,
        message=(
            f"Payment request: {amount} USDC to {recipient} — reason: '{reason}'. "
            f"Investigate the recipient and produce a risk assessment."
        ),
        max_turns=6,
    )


if __name__ == "__main__":
    run_payment_approval(
        recipient="alice.fetch",
        amount=50.0,
        reason="research report delivery",
    )
