"""Optional scrape provider adapters.

Firecrawl remains the active default. Swap provider construction in
`interviewbuddy.settings.Settings.scrape_provider()` if Tavily or Apify should
take over when Firecrawl limits or extraction quality become a problem.
"""
