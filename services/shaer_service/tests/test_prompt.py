from services.shaer_service.app.prompt import DEFAULT_ERA, DEFAULT_POET, build_messages
from services.shaer_service.app.schemas import BaytGenerationRequest


def test_build_messages_formats_previous_verses():
    request = BaytGenerationRequest(
        poem_meter="البسيط",
        poem_description="قصيدة عن الشوق.",
        num_verses=3,
        sequence_number=2,
        previous_verses=["بيت أول"],
    )

    messages = build_messages(request)

    assert messages[0]["role"] == "system"
    assert "البسيط" in messages[1]["content"]
    assert "بيت" in messages[1]["content"]
    assert str(request.sequence_number) in messages[1]["content"]


def test_build_messages_fills_defaults():
    request = BaytGenerationRequest(
        poem_meter="الطويل",
        poem_description="عن الشجاعة.",
        num_verses=1,
        sequence_number=1,
    )

    content = build_messages(request)[1]["content"]

    assert DEFAULT_ERA in content
    assert DEFAULT_POET in content
    assert "لا توجد أبيات سابقة" in content
