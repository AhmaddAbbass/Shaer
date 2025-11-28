from services.common_schemas.schemas import PoemSpec, RagSearchHit
from services.yehia_service.app.prompt import (
    build_chat_messages,
    build_feedback_messages,
    build_spec_messages,
    FEEDBACK_SYSTEM_PROMPT,
    SPEC_SYSTEM_PROMPT,
)
from services.common_schemas.schemas import LLMMessage


def test_build_spec_messages_includes_query_and_hits():
    hits = [
        RagSearchHit(
            poem_id="1",
            poem_title=None,
            poet_name="Poet",
            poem_description="desc",
            poem_meter="meter",
            poem_era=None,
            poem_theme=None,
            has_bad_description=False,
        )
    ]
    messages = build_spec_messages("user query", hits)
    assert messages[0].role == "system"
    assert "JSON" in SPEC_SYSTEM_PROMPT
    assert "user query" in messages[1].content
    assert "Poet" in messages[1].content


def test_build_feedback_messages_includes_spec_and_verse():
    spec = PoemSpec(
        poem_meter="meter",
        poem_description="desc",
        num_verses=2,
    )
    messages = build_feedback_messages("bayt text", spec)
    assert messages[0].role == "system"
    assert "JSON" in FEEDBACK_SYSTEM_PROMPT
    assert "bayt text" in messages[1].content
    assert "meter" in messages[1].content


def test_build_chat_messages_passthrough():
    msgs = [LLMMessage(role="user", content="hi")]
    out = build_chat_messages(msgs)
    assert out == msgs
