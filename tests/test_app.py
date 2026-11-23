"""Headless UI test of the Streamlit app in offline demo mode."""

from streamlit.testing.v1 import AppTest


def test_app_runs_offline_research_end_to_end(monkeypatch):
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, "")
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert at.toggle[0].value is True  # demo mode on when the server has no keys

    at.text_area(key="question").input("Is solar cheaper than nuclear power?")
    at.button[-1].click().run()  # the form's "Research" submit button
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]

    text = " ".join(m.value for m in at.markdown)
    assert "## Sources" in text and "[1]" in text
    assert at.metric[2].value != "0"  # sources cited
