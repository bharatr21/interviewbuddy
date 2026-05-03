from interviewbuddy.sources import DEFAULT_SOURCES, find_source, get_source


def test_default_registry_includes_doordash_engineering_blog():
    source = get_source("doordash", DEFAULT_SOURCES)

    assert source.company == "DoorDash"
    assert source.url == "https://careersatdoordash.com/engineering-blog/"
    assert source.kind == "engineering_blog"
    assert source.enabled is True


def test_find_source_accepts_company_name():
    source = find_source("DoorDash", DEFAULT_SOURCES)

    assert source.slug == "doordash"
