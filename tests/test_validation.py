import pytest

from utils.validation import parse_channel_input, parse_many

CID = "UCBJycsmduvYEL83R_U4JriQ"


@pytest.mark.parametrize("text,kind,value", [
    (CID, "id", CID),
    (f"https://www.youtube.com/channel/{CID}", "id", CID),
    (f"youtube.com/channel/{CID}/videos", "id", CID),
    ("@GoogleDevelopers", "handle", "@GoogleDevelopers"),
    ("https://www.youtube.com/@mkbhd/videos", "handle", "@mkbhd"),
    ("https://youtube.com/user/oldname", "username", "oldname"),
    ("https://www.youtube.com/c/CustomName", "custom", "CustomName"),
    ("plainname", "handle", "@plainname"),
])
def test_valid_inputs(text, kind, value):
    p = parse_channel_input(text)
    assert (p.kind, p.value) == (kind, value)


@pytest.mark.parametrize("text", ["", "   ", None, "a", "https://youtu.be/abc123", "https://www.youtube.com/watch?v=xyz",
                                  "https://example.com/@someone", "https://www.youtube.com/channel/UCshort", "@a!"])
def test_invalid_inputs(text):
    with pytest.raises(ValueError):
        parse_channel_input(text)


def test_parse_many_dedupes_limits_and_reports_errors():
    valid, errors = parse_many(["@aaa", "@AAA", "", "https://youtu.be/x", "@bbb", "@ccc", "@ddd", "@eee", "@fff"], limit=5)
    assert [v.value for v in valid] == ["@aaa", "@bbb", "@ccc", "@ddd", "@eee"]
    assert any("video link" in e for e in errors)
    assert any("Only the first 5" in e for e in errors)
