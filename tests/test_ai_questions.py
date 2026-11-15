import json
from types import SimpleNamespace

import pytest

from ai_questions import AIError, QuestionWriter, SYSTEM
from engine import ROUNDS
from web import create_app


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")      # کلید جعلی؛ هیچ درخواست واقعی فرستاده نمی‌شود


def item(i=0, **extra):
    base = {"text": f"سؤال {i}", "options": ["الف", "ب", "ج", "د"], "correct": "B", "is_double": False}
    return {**base, **extra}


def reply(items, finish="STOP"):
    """شبیه شیء جوابِ google-genai: .text و .candidates[0].finish_reason.name"""
    return SimpleNamespace(
        text=json.dumps({"questions": items}),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
    )


class FakeGenai:
    """جای genai.Client؛ بدون شبکه و بدون هزینه. نتیجه‌ها به ترتیب برمی‌گردند (آخری تکرار می‌شود)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def writer(fake, **kw):
    return QuestionWriter(model="test-model", client=fake, **kw)


def good(n=ROUNDS + 2):
    return reply([item(i, is_double=(i == 0)) for i in range(n)])


def test_generates_validated_questions_and_keeps_topic_out_of_system():
    fake = FakeGenai(good())
    questions = writer(fake).generate("فضا")
    assert len(questions) == ROUNDS                      # اضافی بریده می‌شود
    assert questions[0].power_up == "double" and questions[1].power_up is None
    call = fake.calls[0]
    assert call.model == "test-model"
    assert "فضا" in call.contents                         # موضوع در پیام کاربر است
    assert "فضا" not in call.config.system_instruction    # نه در قواعد ثابت
    assert call.config.system_instruction == SYSTEM
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_json_schema["required"] == ["questions"]


def test_asks_for_spare_questions():
    fake = FakeGenai(good())
    writer(fake).generate("فضا")
    assert str(ROUNDS + 2) in fake.calls[0].contents


@pytest.mark.parametrize("bad", [
    item(options=["a", "b", "c"]),            # ۳ گزینه
    item(correct="E"),                        # جواب خارج از A تا D
    {"text": "ناقص"},                          # فیلد کم
])
def test_invalid_ai_output_is_rejected(bad):
    with pytest.raises(AIError):
        writer(FakeGenai(reply([bad] * ROUNDS))).generate("فضا")


def test_bad_questions_are_dropped_not_fatal():
    dup = item(0, options=["مریخ", "زهره", "مریخ", "عطارد"])     # گزینه‌ی تکراری
    broken = {"text": "ناقص"}
    spare = [item(i) for i in range(1, ROUNDS + 1)]
    questions = writer(FakeGenai(reply([dup, broken] + spare))).generate("فضا")
    assert len(questions) == ROUNDS and all(len(set(q.options)) == 4 for q in questions)


def test_duplicate_options_without_spares_fail():
    dup = item(0, options=["مریخ", "زهره", "مریخ", "عطارد"])
    with pytest.raises(AIError):
        writer(FakeGenai(reply([dup] + [item(i) for i in range(1, ROUNDS)]))).generate("فضا")   # فقط n-1 سؤال سالم


def test_too_few_questions():
    with pytest.raises(AIError):
        writer(FakeGenai(reply([item()]))).generate("فضا")


def test_non_json_output_and_empty_text():
    bad = reply([item()] * ROUNDS)
    bad.text = "نه JSON"
    with pytest.raises(AIError):
        writer(FakeGenai(bad)).generate("فضا")
    bad.text = None
    with pytest.raises(AIError):
        writer(FakeGenai(bad)).generate("فضا")


def test_blocked_truncated_or_empty_candidates():
    for finish in ("SAFETY", "MAX_TOKENS"):
        with pytest.raises(AIError):
            writer(FakeGenai(reply([item()] * ROUNDS, finish=finish))).generate("x")
    empty = SimpleNamespace(text=None, candidates=[])             # مسدود شدن کامل prompt
    with pytest.raises(AIError):
        writer(FakeGenai(empty)).generate("x")


def test_missing_key_and_invalid_model_never_call_the_api(monkeypatch):
    fake = FakeGenai(good())
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(AIError):
        writer(fake).generate("x")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(AIError, match="نامعتبر"):
        QuestionWriter(model="نام-مدل", client=fake).generate("x")      # جایگاه‌نمای فارسی
    assert not fake.calls


def test_failure_cause_is_logged_not_shown(caplog):
    fake = FakeGenai(RuntimeError("404 NOT_FOUND: model is no longer available"))
    with caplog.at_level("WARNING", logger="quiz.ai"):
        with pytest.raises(AIError) as raised:
            writer(fake).generate("x")
    assert "404" in caplog.text and "404" not in str(raised.value)


def test_hourly_budget():
    now = [0.0]
    w = writer(FakeGenai(good()), max_per_hour=2, clock=lambda: now[0])
    w.generate("a")
    w.generate("b")
    with pytest.raises(AIError):
        w.generate("c")
    now[0] += 3601                                       # یک ساعت بعد دوباره مجاز است
    w.generate("d")


# --- از طریق API بازی ---

def make_client(tmp_path, ai_writer):
    app = create_app({
        "TESTING": True, "SECRET_KEY": "t", "LOG_PATH": None, "DB_PATH": None,
        "LEADERBOARD_PATH": tmp_path / "lb.json", "AI_WRITER": ai_writer,
    })
    client = app.test_client()
    token = client.get("/api/state").json["csrf_token"]
    return client, {"X-CSRF-Token": token}


def test_match_uses_ai_questions(tmp_path):
    ai = writer(FakeGenai(reply([item(i, text=f"AI سؤال {i}") for i in range(ROUNDS + 2)])))
    client, headers = make_client(tmp_path, ai)
    response = client.post("/api/matches", json={"players": ["سارا", "علی"], "topic": "فضا"}, headers=headers)
    assert response.status_code == 201
    match = response.json["match"]
    assert match["topic"] == "فضا"
    started = client.post("/api/match/start", json={"turn": match["turn"]}, headers=headers)
    assert started.json["match"]["question"]["text"] == "AI سؤال 0"


def test_ai_failure_is_a_clean_503(tmp_path):
    client, headers = make_client(tmp_path, writer(FakeGenai(RuntimeError("500 boom"))))
    response = client.post("/api/matches", json={"players": ["سارا", "علی"], "topic": "فضا"}, headers=headers)
    assert response.status_code == 503
    assert response.json["error"]["code"] == "ai_unavailable"


def test_topic_without_ai_configured(tmp_path):
    client, headers = make_client(tmp_path, None)
    response = client.post("/api/matches", json={"players": ["سارا", "علی"], "topic": "فضا"}, headers=headers)
    assert response.status_code == 503 and response.json["error"]["code"] == "ai_disabled"


def test_topic_validation_and_default_still_works(tmp_path):
    client, headers = make_client(tmp_path, None)
    too_long = client.post("/api/matches", json={"players": ["سارا", "علی"], "topic": "x" * 61}, headers=headers)
    assert too_long.status_code == 400
    plain = client.post("/api/matches", json={"players": ["سارا", "علی"]}, headers=headers)
    assert plain.status_code == 201 and "topic" not in plain.json["match"]
