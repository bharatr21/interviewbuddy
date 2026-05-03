from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    slug: str
    company: str
    url: str
    kind: str = "engineering_blog"
    enabled: bool = True


DEFAULT_SOURCES: tuple[Source, ...] = (
    Source("openai", "OpenAI", "https://openai.com/research/index/", "research_blog"),
    Source("anthropic", "Anthropic", "https://www.anthropic.com/engineering"),
    Source("nvidia", "NVIDIA", "https://developer.nvidia.com/blog", "technical_blog"),
    Source("microsoft", "Microsoft", "https://devblogs.microsoft.com/engineering-at-microsoft/"),
    Source("uber", "Uber", "https://www.uber.com/blog/engineering"),
    Source("lyft", "Lyft", "https://eng.lyft.com/"),
    Source("pinterest", "Pinterest", "https://medium.com/pinterest-engineering"),
    Source("linkedin", "LinkedIn", "https://www.linkedin.com/blog/engineering"),
    Source("salesforce", "Salesforce", "https://engineering.salesforce.com/"),
    Source("servicenow", "ServiceNow", "https://www.servicenow.com/blogs.html", "technical_blog"),
    Source("meta", "Meta", "https://engineering.fb.com/"),
    Source("apple", "Apple", "https://machinelearning.apple.com/", "research_blog"),
    Source("amazon", "Amazon", "https://www.amazon.science/blog/", "research_blog"),
    Source("netflix", "Netflix", "https://netflixtechblog.com/", "technical_blog"),
    Source("google", "Google", "https://research.google/blog/", "research_blog"),
    Source("doordash", "DoorDash", "https://careersatdoordash.com/engineering-blog/"),
)


def get_source(slug: str, sources: tuple[Source, ...] = DEFAULT_SOURCES) -> Source:
    normalized = slug.strip().lower()
    for source in sources:
        if source.slug == normalized:
            return source
    raise KeyError(f"Unknown source: {slug}")


def find_source(value: str, sources: tuple[Source, ...] = DEFAULT_SOURCES) -> Source:
    normalized = _normalize_source_key(value)
    for source in sources:
        if source.slug == normalized or _normalize_source_key(source.company) == normalized:
            return source
    raise KeyError(f"Unknown source or company: {value}")


def _normalize_source_key(value: str) -> str:
    return "".join(character for character in value.strip().lower() if character.isalnum())
