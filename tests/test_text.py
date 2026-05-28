from aibias import TextBiasDetector


def test_slur_high_severity():
    r = TextBiasDetector().analyze("He called him a faggot during the meeting.")
    assert any(f.severity == "high" and f.kind == "slur_or_pejorative" for f in r.findings)
    assert "faggot" in r.slurs_detected


def test_no_bias_text_runs_clean():
    text = "The committee reviewed the proposal and voted unanimously."
    r = TextBiasDetector().analyze(text)
    assert r.bias_index["overall"] <= 25


def test_gender_occupation_stereotype():
    text = (
        "The engineer fixed the server. He had been a soldier and is now "
        "a programmer. His wife, a nurse, works as a secretary part-time."
    )
    r = TextBiasDetector().analyze(text)
    kinds = {f.kind for f in r.findings}
    assert "occupational_stereotype" in kinds


def test_report_serializable():
    r = TextBiasDetector().analyze("Hello world.")
    import json
    json.dumps(r.to_dict())  # must not raise
