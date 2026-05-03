from __future__ import annotations

import streamlit as st

from interviewbuddy.corpus import JsonlCorpusStore
from interviewbuddy.paths import DEFAULT_CORPUS_PATH
from interviewbuddy.query_crawl_agent import CrawlBudget, QueryCrawlAgent, QueryCrawlRequest
from interviewbuddy.rag import CoachAnswer
from interviewbuddy.service import build_corpus_agent
from interviewbuddy.settings import get_settings
from interviewbuddy.sources import DEFAULT_SOURCES, find_source


st.set_page_config(page_title="Interview Buddy", page_icon="IB")
st.title("Interview Buddy")
st.caption(f"Corpus: `{DEFAULT_CORPUS_PATH}`. Run `uv run interviewbuddy ingest doordash --limit 5` to add live Firecrawl content.")

documents = []
try:
    documents = JsonlCorpusStore(DEFAULT_CORPUS_PATH).load()
except Exception:
    documents = []

with st.sidebar:
    st.header("Corpus")
    st.metric("Documents", len(documents))
    company_counts: dict[str, int] = {}
    for document in documents:
        company_counts[document.company] = company_counts.get(document.company, 0) + 1
    for company_name, count in sorted(company_counts.items()):
        st.caption(f"{company_name}: {count}")

companies = ["All companies", *[source.company for source in DEFAULT_SOURCES]]
company = st.selectbox("Company filter", companies)
limit = st.slider("Retrieved snippets", min_value=1, max_value=8, value=4)
auto_crawl = st.checkbox("Auto-crawl if evidence is weak", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a system design or company-specific prep question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    selected_company = None if company == "All companies" else company
    settings = get_settings()
    crawl_report = None
    if auto_crawl:
        if selected_company is None:
            answer_message = "Choose a company before enabling auto-crawl so crawling stays source-allowlisted."
            answer = CoachAnswer(message=answer_message, citations=[])
        else:
            try:
                source = find_source(selected_company)
                result = QueryCrawlAgent(
                    source=source,
                    provider=settings.scrape_provider(),
                    corpus_store=JsonlCorpusStore(DEFAULT_CORPUS_PATH),
                    vector_store=settings.vector_store(),
                    embedding_provider=settings.embedding_provider(),
                    chat_provider=settings.chat_provider(),
                ).run(
                    QueryCrawlRequest(
                        question=prompt,
                        company=source.company,
                        limit=limit,
                        budget=CrawlBudget(),
                    )
                )
                answer = result.answer
                crawl_report = result.crawl_report
            except ValueError as error:
                answer = CoachAnswer(
                    message=(
                        f"{error}\n\n"
                        "Add it to `.env`, then restart the Streamlit app. "
                        "You can still turn off auto-crawl and ask against the existing corpus."
                    ),
                    citations=[],
                )
    else:
        answer = build_corpus_agent(
            DEFAULT_CORPUS_PATH,
            embedding_provider=settings.embedding_provider(),
            chat_provider=settings.chat_provider(),
            vector_store=settings.vector_store() if DEFAULT_CORPUS_PATH.exists() else None,
        ).answer(prompt, company=selected_company, limit=limit)
    st.session_state.messages.append({"role": "assistant", "content": answer.message})

    with st.chat_message("assistant"):
        st.markdown(answer.message)
        if crawl_report:
            st.caption(
                "Auto-crawl: "
                f"used_existing={crawl_report.used_existing_evidence}, "
                f"accepted={crawl_report.accepted_count}, "
                f"skipped={crawl_report.skipped_count}, "
                f"failed={crawl_report.failed_count}, "
                f"provider_calls={crawl_report.provider_call_count}"
            )
        for citation in answer.citations:
            with st.expander(f"{citation.company}: {citation.title}"):
                st.write(citation.snippet)
                st.caption(f"{citation.url} | score {citation.score:.3f}")
