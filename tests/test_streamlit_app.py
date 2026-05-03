from streamlit.testing.v1 import AppTest

from interviewbuddy.settings import Settings


def test_streamlit_auto_crawl_missing_firecrawl_key_shows_setup_message(monkeypatch):
    def missing_firecrawl_key(self):
        raise ValueError("FIRECRAWL_API_KEY is required")

    monkeypatch.setattr(Settings, "scrape_provider", missing_firecrawl_key)

    app = AppTest.from_file("src/interviewbuddy/streamlit_app.py")
    app.run()
    app.selectbox[0].select("DoorDash").run()
    app.checkbox[0].check().run()
    app.chat_input[0].set_value("How does DoorDash approach reliability?").run(timeout=10)

    assert len(app.exception) == 0
    assert any("FIRECRAWL_API_KEY is required" in message.value for message in app.markdown)
