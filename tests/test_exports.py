import io
import json
import re
from pathlib import Path

import openpyxl
import pandas as pd
from docx import Document

from tests.conftest import goto
from src.ai_provider import AIProvider, AIProviderError, parse_json_loose
from utils.data_processing import SECRET_PATTERNS, df_to_csv_bytes, redact_secrets, results_to_json_bytes, to_jsonable
from utils.export_excel import build_workbook
from utils.export_pdf import build_pdf
from utils.export_word import build_docx

FAKE_KEY = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY12345"


def test_pdf_is_valid_and_labels_sample_data(results):
    pdf = build_pdf(results)
    assert pdf.startswith(b"%PDF-") and len(pdf) > 20_000 and pdf.rstrip().endswith(b"%%EOF")


def test_excel_sheets_contents_filters_and_links(results):
    wb = openpyxl.load_workbook(io.BytesIO(build_workbook(results)))
    assert wb.sheetnames == ["Channel Overview", "Growth Roadmap", "Roadmap Actions", "Daily Plan", "Video Performance", "Title Analysis",
                             "Title Patterns", "Topic Analysis", "Competitors", "Auto Competitors", "Content Gaps", "Content Calendar",
                             "Sources and Notes"]
    vp = wb["Video Performance"]
    headers = [c.value for c in vp[1]]
    assert vp.max_row == len(results["videos"]) + 1 and vp.auto_filter.ref and vp.freeze_panes == "A2"
    url_col = headers.index("Video URL") + 1
    assert vp.cell(row=2, column=url_col).hyperlink is not None
    views_col = headers.index("Views (public)") + 1
    assert vp.cell(row=2, column=views_col).number_format == "#,##0"
    assert wb["Content Gaps"].max_row == len(results["gaps"]) + 1
    notes = " ".join(str(c.value) for row in wb["Sources and Notes"].iter_rows() for c in row if c.value)
    assert "SAMPLE DATA" in notes and "YouTube Data API v3" in notes


def test_excel_only_includes_relevant_sheets(now):
    from src.pipeline import run_analysis
    from src.sample_data import SampleClient
    r = run_analysis(SampleClient(now), "@sample_own", [], sample_size=15, now=now)
    names = openpyxl.load_workbook(io.BytesIO(build_workbook(r))).sheetnames
    assert "Competitors" not in names and "Content Calendar" not in names and "Video Performance" in names


def test_word_report_opens_and_has_sections(results):
    doc = Document(io.BytesIO(build_docx(results)))
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert any("Content gap" in h for h in headings) and len(doc.tables) >= 4


def test_csv_encoding_roundtrip_with_unicode():
    df = pd.DataFrame({"title": ["Café ☕ — naïve 日本語", "plain"], "tags": [["a", "b"], []],
                       "published_at": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z"], utc=True)})
    raw = df_to_csv_bytes(df)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8
    back = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert back.loc[0, "title"] == "Café ☕ — naïve 日本語" and back.loc[0, "tags"] == "a, b"


def test_json_is_valid_complete_and_nan_safe(results):
    payload = json.loads(results_to_json_bytes(results))  # raises if NaN/Infinity or bad types slipped through
    assert payload["meta"]["mode"].startswith("SAMPLE") and len(payload["videos"]) == len(results["videos"])
    assert "limitations" in payload and "gaps" in payload and "NaN" not in results_to_json_bytes(results).decode()


def test_to_jsonable_handles_numpy_nan_timestamps():
    import numpy as np
    out = to_jsonable({"a": np.int64(3), "b": float("nan"), "c": pd.Timestamp("2026-01-01", tz="UTC"), "d": pd.NaT, "e": np.float64(1.5)})
    assert out == {"a": 3, "b": None, "c": "2026-01-01T00:00:00+00:00", "d": None, "e": 1.5}


def test_no_secrets_in_any_export(results):
    tainted = dict(results)
    tainted["notes"] = list(results["notes"]) + [f"debug leaked {FAKE_KEY}"]
    blob = results_to_json_bytes(tainted).decode()
    assert FAKE_KEY not in blob and "[REDACTED]" in blob
    for name, data in (("pdf", build_pdf(results)), ("xlsx", build_workbook(results)), ("docx", build_docx(results)),
                       ("json", results_to_json_bytes(results))):
        text = data.decode("latin-1")
        assert not any(p.search(text) for p in SECRET_PATTERNS), name


def test_source_tree_contains_no_hardcoded_keys():
    root = Path(__file__).resolve().parents[1]
    for path in list(root.rglob("*.py")) + list(root.rglob("*.toml*")) + [root / "README.md"]:
        if not path.exists():
            continue
        if "tests" in path.parts or path.name == "secrets.toml":
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"AIza[0-9A-Za-z_\-]{35}", text), path
        assert not re.search(r"sk-[A-Za-z0-9_\-]{20,}", text), path
    ignore = (root / ".gitignore").read_text()
    assert ".env" in ignore and ".streamlit/secrets.toml" in ignore  # local key file must never be committed


def test_redact_secrets():
    assert FAKE_KEY not in redact_secrets(f"url?key={FAKE_KEY}&x=1")


# ------------------------------------------------------------------ AI provider
class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body

    def json(self):
        return self._b


class FakeSession:
    def __init__(self, resp):
        self.resp, self.last = resp, None

    def post(self, url, headers=None, json=None, timeout=None):
        self.last = (url, headers, json)
        return self.resp


def test_ai_provider_anthropic_and_openai_parsing_and_missing_key():
    a = AIProvider("anthropic", "k", session=FakeSession(FakeResp(200, {"content": [{"type": "text", "text": '```json\n[{"x": 1}]\n```'}]})))
    assert a.complete_json("s", "u") == [{"x": 1}] and a.session.last[1]["x-api-key"] == "k"
    o = AIProvider("openai", "k", session=FakeSession(FakeResp(200, {"choices": [{"message": {"content": 'Sure! [1, 2]'}}]})))
    assert o.complete_json("s", "u") == [1, 2]
    assert not AIProvider("anthropic", "").is_configured
    try:
        AIProvider("anthropic", "").complete("s", "u")
        raise AssertionError("expected AIProviderError")
    except AIProviderError:
        pass


def test_ai_provider_errors_do_not_leak_key():
    p = AIProvider("anthropic", "SECRETKEY123", session=FakeSession(FakeResp(401, {"error": {"message": "bad key SECRETKEY123"}})))
    try:
        p.complete("s", "u")
        raise AssertionError("expected error")
    except AIProviderError as e:
        assert "SECRETKEY123" not in str(e)


def test_parse_json_loose_rejects_garbage():
    try:
        parse_json_loose("no json here")
        raise AssertionError("expected error")
    except AIProviderError:
        pass


# ------------------------------------------------------------------ Streamlit behaviour
def _run_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
    return at


def _analyse(at, channel):
    [t for t in at.sidebar.text_input if "Your channel" in t.label][0].set_value(channel)
    at.sidebar.button[0].click()
    return at.run()


def test_app_renders_empty_state_without_errors():
    at = _run_app()
    assert not at.exception and "results" not in at.session_state


def test_app_has_no_key_or_competitor_inputs_and_no_search():
    at = _run_app()
    labels = [w.label for w in list(at.sidebar.text_input) + list(at.sidebar.text_area)]
    assert labels == ["Your channel"]  # no API-key prompts, no competitor box
    assert not at.exception


def test_app_invalid_input_shows_friendly_error():
    at = _analyse(_run_app(), "https://youtu.be/notachannel")
    assert not at.exception and any("Input problem" in e.value for e in at.error)


def test_app_tabs_include_roadmap_and_no_search():
    at = _analyse(_run_app(), "@sample_own")
    names = list(at.radio(key="active_tab").options)
    assert any("Growth Roadmap" in n for n in names) and "Search" not in names and not at.exception
    assert len(at.session_state["results"]["competitors"]) == 11  # you + 10 auto-discovered


def test_app_downloads_survive_reruns_and_new_search_replaces_results():
    at = _analyse(_run_app(), "@sample_own")
    assert not at.exception and at.session_state["results"]["channel"]["title"].endswith("Sample_Own")
    goto(at, "Sources and Export")
    [b for b in at.button if b.label == "Build / refresh export files"][0].click()
    at.run()
    built = {k: len(v) for k, v in at.session_state["exports"].items()}
    assert set(built) == {"pdf", "xlsx", "docx", "json"} and all(built.values())
    at.run()  # an unrelated rerun (as caused by clicking any download button)
    assert {k: len(v) for k, v in at.session_state["exports"].items()} == built
    first_title = at.session_state["results"]["channel"]["title"]
    at = _analyse(at, "@another_sample")
    assert at.session_state["results"]["channel"]["title"] != first_title
    assert "exports" not in at.session_state or not at.session_state["exports"]  # old exports discarded
    assert at.session_state["results"]["meta"]["channel_input"] == "@another_sample"  # nothing carried over from the previous run


def test_app_calendar_marks_exports_stale():
    at = _analyse(_run_app(), "@sample_own")
    goto(at, "Sources and Export")
    [b for b in at.button if b.label == "Build / refresh export files"][0].click()
    at.run()
    goto(at, "AI Content Plan")
    [b for b in at.button if b.label == "Generate content calendar"][0].click()
    at.run()
    goto(at, "Sources and Export")
    assert not at.exception and len(at.session_state["results"]["calendar"]) > 0
    assert any("Results changed" in w.value for w in at.warning)
