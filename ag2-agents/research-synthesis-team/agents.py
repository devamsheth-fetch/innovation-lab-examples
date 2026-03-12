"""
AG2 (formerly AutoGen) research synthesis team.
Four specialists collaborate under GroupChat with LLM-driven speaker selection.
"""
from autogen import AssistantAgent, LLMConfig

def build_agents(llm_config: LLMConfig) -> list[AssistantAgent]:
    web_researcher = AssistantAgent(
        name="web_researcher",
        system_message=(
            "You are a research specialist. Search and gather comprehensive information "
            "on the assigned topic using available tools. Cite sources clearly."
        ),
        llm_config=llm_config,
    )
    financial_analyst = AssistantAgent(
        name="financial_analyst",
        system_message=(
            "You are a financial analyst. Analyse market data, trends, and economic "
            "implications of the research topic. Be quantitative when possible."
        ),
        llm_config=llm_config,
    )
    tech_analyst = AssistantAgent(
        name="tech_analyst",
        system_message=(
            "You are a technology analyst. Evaluate technical aspects, feasibility, "
            "and innovation potential of the research topic."
        ),
        llm_config=llm_config,
    )
    synthesizer = AssistantAgent(
        name="synthesizer",
        system_message=(
            "You are a synthesis expert. Once all specialists have contributed, "
            "produce a final structured report combining all perspectives. "
            "Format as Markdown with sections: Summary, Financial Analysis, "
            "Technical Analysis, Conclusions. End with TERMINATE."
        ),
        llm_config=llm_config,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    return [web_researcher, financial_analyst, tech_analyst, synthesizer]
