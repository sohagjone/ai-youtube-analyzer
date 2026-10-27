"""AI YouTube Channel Analyzer - Streamlit app (public YouTube data + optional AI)."""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src import content_gap_analysis as gaps_a
from src import content_strategy as strat
from src import title_analysis as ta
from src.ai_provider import AIProvider, AIProviderError
from src.pipeline import apply_manual_topics, recompute_topics, run_analysis
from src.sample_data import SampleClient
from src.topic_analysis import parse_topic_mapping
from src.youtube_api import (InvalidAPIKeyError, MissingAPIKeyError, QuotaExceededError, YouTubeAPIError,
                             YouTubeClient, estimate_quota)
from utils.data_processing import df_to_csv_bytes, format_duration, format_number, results_to_json_bytes
from utils.export_excel import build_workbook
from utils.export_pdf import build_pdf
from utils.export_word import build_docx

st.set_page_config(page_title="AI YouTube Channel Analyzer", page_icon="📊", layout="wide")
RED = ["#d62828", "#8d99ae", "#f77f00", "#003049", "#6a994e"]


# --------------------------------------------------------------------------- helpers
def get_secret(name: str) -> str:
    """Secrets file (Streamlit Cloud) first, then environment variables. Never printed or exported."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:  # noqa: BLE001 - no secrets file configured
        pass
    return os.environ.get(name, "")


def youtube_key() -> str:
    return st.session_state.get("session_yt_key") or get_secret("YOUTUBE_API_KEY")


def ai_provider() -> AIProvider | None:
    cfg = {k: get_secret(k) for k in ("AI_PROVIDER", "AI_API_KEY", "AI_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    if st.session_state.get("session_ai_key"):
        cfg["AI_API_KEY"] = st.session_state["session_ai_key"]
    try:
        prov = AIProvider.from_config(cfg)
    except AIProviderError:
        return None
    return prov if prov.is_configured else None


def show_df(df: pd.DataFrame, link_cols: tuple[str, ...] = ("url", "top_video_url"), height: int | None = None) -> None:
    cfg = {c: st.column_config.LinkColumn(c.replace("_", " ").title(), display_text="open") for c in link_cols if c in df.columns}
    for c in df.columns:
        if str(c).startswith(("share", "short_form", "like_rate", "comment_rate", "engagement")) or c in ("recent_share", "older_share"):
            cfg[c] = st.column_config.NumberColumn(c.replace("_", " ").title(), format="percent")
    kwargs = {"height": height} if height else {}
    st.dataframe(df, column_config=cfg, hide_index=True, width="stretch", **kwargs)


def csv_button(label: str, df: pd.DataFrame, name: str, key: str) -> None:
    if df is not None and len(df):
        st.download_button(label, df_to_csv_bytes(df), file_name=name, mime="text/csv", key=key)


def reset_analysis() -> None:
    for k in ("results", "exports", "exports_version", "search_results", "tpl_text"):
        st.session_state.pop(k, None)


def metric(col, label: str, value, help_text: str | None = None) -> None:
    col.metric(label, value, help=help_text)


# --------------------------------------------------------------------------- sidebar
def sidebar() -> dict | None:
    st.sidebar.title("📊 Channel Analyzer")
    key_ok = bool(youtube_key())
    with st.sidebar.expander("API configuration", expanded=not key_ok):
        st.caption("Keys are read from Streamlit secrets or environment variables. You may also paste a key for this browser session only (never stored or exported).")
        st.text_input("YouTube Data API key (session only)", type="password", key="session_yt_key")
        st.text_input("AI provider key (optional, session only)", type="password", key="session_ai_key")
    st.sidebar.markdown(f"**YouTube API:** {'✅ configured' if key_ok else '⚠️ not configured'}")
    prov = ai_provider()
    st.sidebar.markdown(f"**AI features:** {'✅ ' + prov.provider + ' / ' + prov.model if prov else '➖ off (core analysis still works)'}")

    with st.sidebar.form("analysis_form"):
        channel = st.text_input("Your channel (URL, @handle or channel ID)", placeholder="https://www.youtube.com/@yourchannel")
        competitors = st.text_area("Competitor channels (up to 5, one per line)", height=110, placeholder="@competitor1\nhttps://www.youtube.com/@competitor2")
        sample_size = st.slider("Videos per channel (sample size)", 10, 200, 50, step=5)
        use_dates = st.checkbox("Limit to a date range")
        dr = st.date_input("Date range", value=(date.today() - timedelta(days=365), date.today()), disabled=not use_dates)
        sample_mode = st.checkbox("Use clearly-labelled SAMPLE data (no API key needed)", value=not key_ok,
                                  help="Illustrative fake data for trying the app. Never mix it up with real results.")
        submitted = st.form_submit_button("Run analysis", type="primary")
    if key_ok and not sample_mode:
        n_comp = len([c for c in competitors.splitlines() if c.strip()][:5])
        st.sidebar.caption(f"Estimated API quota for this run: about {estimate_quota(sample_size, 1 + n_comp)} of 10,000 daily units.")
    st.sidebar.button("Start a new analysis", on_click=reset_analysis)
    if not submitted:
        return None
    date_from = date_to = None
    if use_dates and isinstance(dr, (tuple, list)) and len(dr) == 2:
        date_from = datetime.combine(dr[0], time.min, tzinfo=timezone.utc)
        date_to = datetime.combine(dr[1], time.max, tzinfo=timezone.utc)
    return {"channel": channel, "competitors": competitors.splitlines(), "sample_size": sample_size,
            "date_from": date_from, "date_to": date_to, "sample_mode": sample_mode}


def run_from_form(form: dict) -> None:
    reset_analysis()  # a new analysis never mixes with earlier results
    try:
        if form["sample_mode"]:
            client = SampleClient()
        else:
            client = YouTubeClient(youtube_key())
        bar = st.progress(0.0, text="Starting...")
        results = run_analysis(client, form["channel"], form["competitors"], form["sample_size"], form["date_from"], form["date_to"],
                               progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        bar.empty()
        st.session_state["results"] = results
    except ValueError as exc:
        st.error(f"Input problem: {exc}")
    except MissingAPIKeyError:
        st.error("No YouTube API key is configured. Add one in the sidebar/secrets, or tick the SAMPLE-data option to try the app.")
    except InvalidAPIKeyError as exc:
        st.error(str(exc))
    except QuotaExceededError as exc:
        st.error(f"{exc} You can switch on SAMPLE-data mode to explore the interface meanwhile.")
    except YouTubeAPIError as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- tabs
def tab_overview(r: dict) -> None:
    life, samp, freq = r["summary"]["lifetime"], r["summary"]["sample"], r["summary"]["upload_frequency"]
    st.subheader(life["channel_name"])
    if str(r["meta"]["mode"]).startswith("SAMPLE"):
        st.warning("SAMPLE DATA: everything below is illustrative and is not real YouTube data.")
    for n in r["notes"]:
        if not n.startswith("SAMPLE") and ": SAMPLE" not in n:
            st.info(n)
    for w in r["warnings"]:
        st.warning(w)
    st.markdown("##### Lifetime channel statistics (whole channel, public)")
    c = st.columns(4)
    metric(c[0], "Subscribers", "Hidden" if life["subscribers_hidden"] else format_number(life["subscribers"]), "Public count; YouTube rounds large numbers.")
    metric(c[1], "Total views", format_number(life["total_views"]))
    metric(c[2], "Total videos", format_number(life["total_videos"]))
    metric(c[3], "Lifetime avg views / video", format_number(life["lifetime_avg_views_per_video"]), "Total views divided by total videos.")
    st.markdown(f"##### Selected sample ({samp.get('videos_in_sample', 0)} videos)")
    if samp.get("videos_in_sample", 0) == 0:
        st.info("No public videos matched the selected sample/date range.")
        return
    c = st.columns(4)
    metric(c[0], "Median views", format_number(samp["median_views"]))
    metric(c[1], "Mean views", format_number(samp["mean_views"]), "The mean is pulled up by outliers; compare with the median.")
    metric(c[2], "Uploads / 30 days", "n/a" if freq["uploads_per_30_days"] is None else f"{freq['uploads_per_30_days']:.1f}")
    metric(c[3], "Short-form share", f"{samp['short_form_share']:.0%}", "Videos of 3 minutes or less (the API has no Shorts flag).")
    df = r["videos"]
    fig = px.bar(df.sort_values("published_at"), x="published_at", y="views", hover_data=["title"], color_discrete_sequence=RED,
                 title="Public views by upload date (selected sample)")
    fig.add_hline(y=samp["median_views"], line_dash="dash", annotation_text="sample median")
    st.plotly_chart(fig, width="stretch")
    pp = r["publishing_patterns"]
    if len(pp["weekday"]):
        left, right = st.columns(2)
        left.plotly_chart(px.bar(pp["weekday"], x="weekday", y="videos", title="Uploads by weekday (UTC)", color_discrete_sequence=RED), width="stretch")
        right.plotly_chart(px.bar(pp["weekday"], x="weekday", y="median_views", title="Median views by upload weekday (small samples are noisy)", color_discrete_sequence=RED), width="stretch")
    with st.expander("Channel description"):
        st.write(life["description"] or "No description.")
    csv_button("Download overview (CSV)", r["overview_table"], "channel_overview.csv", "dl_overview")


def tab_videos(r: dict) -> None:
    df = r["videos"]
    if df.empty:
        st.info("No videos to analyse.")
        return
    with st.expander("Methodology (how each number is calculated)"):
        st.markdown("- **Views per day** = public views / days since publication (min 1 day). Older videos have had longer to accumulate views.\n"
                    "- **Relative to median** = views / median views of this sample.\n"
                    "- **Engagement rate** = (likes + comments) / views using whatever is public; blank when both are hidden.\n"
                    "- **Below typical** = 0.5x median or less *and* older than 30 days; newer videos are marked *too new to judge*.\n"
                    "- Public views are **not** watch time, retention or click-through rate.")
    c1, c2 = st.columns(2)
    fmts = c1.multiselect("Format", sorted(df["format"].unique()), default=sorted(df["format"].unique()))
    topics = c2.multiselect("Topic", sorted(df["topic"].unique()), default=sorted(df["topic"].unique()))
    view = df[df["format"].isin(fmts) & df["topic"].isin(topics)]
    cols = ["title", "url", "published_at", "duration_seconds", "views", "likes", "comments", "views_per_day", "relative_to_median",
            "engagement_rate", "format", "topic", "performance_flag"]
    show_df(view[cols], height=380)
    csv_button("Download video data (CSV)", view[cols], "video_performance.csv", "dl_videos")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(view, x="relative_to_median", nbins=20, title="Distribution: views relative to sample median", color_discrete_sequence=RED), width="stretch")
    b.plotly_chart(px.scatter(view, x="age_days", y="views", hover_data=["title"], title="Views vs video age (days)", color_discrete_sequence=RED, log_y=True), width="stretch")
    left, right = st.columns(2)
    left.markdown("**Most-viewed in sample**")
    left.dataframe(r["top_videos"][["title", "views", "relative_to_median"]].head(5), hide_index=True, width="stretch")
    right.markdown("**Lowest-viewed in sample** (check age before judging)")
    right.dataframe(r["lowest_videos"][["title", "views", "age_days", "performance_flag"]].head(5), hide_index=True, width="stretch")
    if len(r["format_comparison"]):
        st.markdown("**Format comparison**")
        show_df(r["format_comparison"])


def tab_titles(r: dict) -> None:
    df = r["videos"]
    if df.empty:
        st.info("No videos to analyse.")
        return
    st.caption("Patterns are observational. They show what co-occurs with higher or lower views in this sample, not what causes it. Groups under 3 videos are flagged.")
    show_df(r["title_patterns"])
    csv_button("Download title patterns (CSV)", r["title_patterns"], "title_patterns.csv", "dl_titlepat")
    flagged = df[df["wording_flags"] != ""][["title", "wording_flags", "views"]]
    st.markdown("**Wording to review** (possible overpromising; a prompt for your judgement, not a verdict)")
    st.dataframe(flagged, hide_index=True, width="stretch") if len(flagged) else st.write("No flagged wording found.")
    a, b = st.columns(2)
    a.markdown("**Near-duplicate titles**")
    a.dataframe(r["similar_titles"], hide_index=True, width="stretch") if len(r["similar_titles"]) else a.write("None found.")
    b.markdown("**Repeated openings**")
    b.dataframe(r["repeated_structures"], hide_index=True, width="stretch") if len(r["repeated_structures"]) else b.write("None found.")

    st.markdown("---")
    st.markdown("#### Title ideas for a selected video")
    options = df["title"].tolist()
    pick = st.selectbox("Choose a video", options)
    row = df[df["title"] == pick].iloc[0]
    kws = df.attrs.get("title_keywords") or r["videos"].attrs.get("title_keywords") or []
    ideas = ta.suggest_titles_rule_based(pick, kws)
    prov = ai_provider()
    if st.button("Generate AI alternatives", disabled=prov is None, help=None if prov else "Configure an AI key to enable"):
        try:
            with st.spinner("Asking the AI provider..."):
                r["title_suggestions"][pick] = ta.suggest_titles_ai(prov, pick, row["description"])
        except (AIProviderError, ValueError) as exc:
            st.warning(f"AI suggestions unavailable ({exc}); showing rule-based starting points.")
    shown = r["title_suggestions"].get(pick) or ideas
    label = "AI-generated" if pick in r["title_suggestions"] else "Rule-based starting points"
    st.caption(f"{label}: recommendations, not guaranteed improvements. Check each still describes what the video actually delivers.")
    for i, idea in enumerate(shown, 1):
        st.markdown(f"**{i}. {idea['title']}**  \n{idea['improvement']}")

    st.markdown("---")
    st.markdown("#### Thumbnail statistics (optional)")
    st.caption("Simple image statistics from public thumbnails (brightness, contrast, saturation, edge busyness). Text density needs OCR and is not computed. These say nothing about click-through rate.")
    if st.button("Analyse up to 20 thumbnails"):
        def fetch(url: str) -> bytes:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            return resp.content
        with st.spinner("Downloading thumbnails..."):
            r["thumbnails"] = ta.analyze_thumbnails(df, fetch, limit=20)
        if r["thumbnails"].empty:
            st.info("No thumbnails could be analysed (sample data has none, or downloads failed).")
    if len(r["thumbnails"]):
        st.dataframe(r["thumbnails"], hide_index=True, width="stretch")
        st.write("Spread across videos (lower = more consistent):", ta.thumbnail_consistency(r["thumbnails"]))


def tab_topics(r: dict) -> None:
    df = r["videos"]
    if df.empty:
        st.info("No videos to analyse.")
        return
    st.caption(f"Method: {r['meta']['topic_method']}. Based on titles, tags and the start of descriptions, not full video content or transcripts.")
    show_df(r["topics"])
    csv_button("Download topic analysis (CSV)", r["topics"], "topic_analysis.csv", "dl_topics")
    t = r["topics"]
    a, b = st.columns(2)
    a.plotly_chart(px.bar(t, x="topic", y="videos", title="Videos per topic", color_discrete_sequence=RED), width="stretch")
    b.plotly_chart(px.bar(t, x="topic", y="median_views", title="Median views per topic", color_discrete_sequence=RED), width="stretch")
    if len(r["topic_trend"]):
        st.markdown("**Recent vs older content** (topic share in the newer vs older half of the sample)")
        show_df(r["topic_trend"])
    with st.expander("Inspect and correct categories"):
        st.markdown("Define your own topics (one per line, `Topic: keyword, keyword`) or correct individual videos.")
        text = st.text_area("Custom topic keywords", key="tpl_text", height=110, placeholder="Espresso: espresso, shot, crema\nGear: grinder, machine")
        if st.button("Re-group using my keywords"):
            recompute_topics(r, parse_topic_mapping(text) or None)
            st.rerun()
        edit = df[["video_id", "title", "topic"]].copy()
        edited = st.data_editor(edit, hide_index=True, disabled=["video_id", "title"], width="stretch", key="topic_editor")
        if st.button("Apply manual corrections"):
            changes = {a_: b_ for a_, b_, c_ in zip(edit["video_id"], edited["topic"], edit["topic"]) if b_ != c_}
            if changes:
                apply_manual_topics(r, changes)
                st.rerun()


def tab_competitors(r: dict) -> None:
    comp = r["competitors"]
    if comp.empty:
        st.info("Add up to five competitor channels in the sidebar and run the analysis to see a comparison.")
        return
    st.caption("Raw views do not show which creator is better: channels differ in age, audience size, niche and history.")
    show_df(comp, link_cols=("url", "top_video_url"))
    csv_button("Download competitor comparison (CSV)", comp, "competitor_comparison.csv", "dl_comp")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(comp, x="channel", y="median_views", color="role", title="Median views (sample)", color_discrete_sequence=RED), width="stretch")
    b.plotly_chart(px.bar(comp, x="channel", y="median_views_per_1k_subs", color="role", title="Median views per 1,000 subscribers (rough size adjustment)", color_discrete_sequence=RED), width="stretch")
    for line in r["competitor_differences"]:
        st.markdown(f"- {line}")
    if len(r["competitor_top_videos"]):
        st.markdown("**Highest-viewed recent competitor videos**")
        show_df(r["competitor_top_videos"])
    with st.expander("Limitations"):
        from src.competitor_analysis import LIMITATIONS
        for lim in LIMITATIONS:
            st.markdown(f"- {lim}")


def tab_gaps(r: dict) -> None:
    gaps = r["gaps"]
    st.info(gaps_a.DISCLAIMER)
    if gaps.empty:
        st.write("No content-gap ideas could be generated from this sample. Try a larger sample or add competitors.")
        return
    with st.expander("How the priority score works"):
        st.markdown("Priority (0-100) = **30% relevance** (does the topic sit near your existing vocabulary?) + **30% demand evidence** "
                    "(competitor videos' performance vs their own channel medians, and how many there are) + **20% differentiation** "
                    "(how little you cover it) + **20% feasibility** (fit with the format you already make). "
                    "Confidence reflects how many videos and channels support the idea. There is no search-volume data in this score.")
    prov = ai_provider()
    if st.button("Refine top ideas with AI", disabled=prov is None, help=None if prov else "Configure an AI key to enable"):
        with st.spinner("Asking the AI provider..."):
            r["gaps"] = gaps_a.refine_gaps_with_ai(prov, gaps)
            r["meta"]["version"] += 1
        st.rerun()
    show_df(gaps[["rank", "gap_type", "topic", "suggested_title", "format", "priority_score", "confidence"]])
    csv_button("Download content gaps (CSV)", gaps, "content_gaps.csv", "dl_gaps")
    for g in gaps.head(10).itertuples():
        with st.expander(f"#{g.rank} {g.topic} - {g.priority_score} ({g.confidence} confidence)"):
            st.markdown(f"**Type:** {g.gap_type}  \n**Audience:** {g.audience}  \n**Format:** {g.format}  \n**Suggested title:** {g.suggested_title} *(recommendation)*")
            st.markdown(f"**Evidence:** {g.evidence}")
            if g.competing_videos:
                st.text(g.competing_videos)
            st.markdown(f"**Why test it:** {g.why_test}  \n**Outline scaffold:** {g.outline}")
            st.markdown(f"Scores - relevance {g.relevance}, demand evidence {g.demand_evidence}, differentiation {g.differentiation}, feasibility {g.feasibility}")
            st.caption(g.limitations)


def tab_plan(r: dict) -> None:
    st.caption("Every row is tied to evidence from your analysis or clearly labelled a creative hypothesis. Nothing here promises views, subscribers or virality.")
    c1, c2 = st.columns(2)
    niche = c1.text_input("Niche", placeholder="e.g. home coffee brewing")
    audience = c2.text_input("Target audience", placeholder="e.g. beginners buying their first grinder")
    c3, c4, c5 = st.columns(3)
    language = c3.text_input("Language", value="English")
    horizon = c4.radio("Calendar length", [7, 30], horizontal=True, format_func=lambda d: f"{d} days")
    ppw = c5.slider("Posts per week", 1, 7, 3)
    formats = st.multiselect("Formats", strat.FORMATS, default=strat.FORMATS[:3])
    goals = st.multiselect("Goals", list(strat.CTAS), default=["subscribers"])
    prov = ai_provider()
    use_ai = st.checkbox("Refine wording with AI", value=False, disabled=prov is None, help=None if prov else "Configure an AI key to enable")
    if st.button("Generate content calendar", type="primary"):
        r["calendar"] = strat.generate_calendar(r["gaps"], r["topics"], days=horizon, posts_per_week=ppw, formats=formats, niche=niche,
                                                audience=audience, goals=goals, language=language, provider=prov if use_ai else None)
        r["meta"]["version"] += 1
    cal = r["calendar"]
    if len(cal):
        show_df(cal)
        csv_button("Download calendar (CSV)", cal, "content_calendar.csv", "dl_cal")
    else:
        st.info("Generate a calendar to see recommended topics, titles, hooks, talking points and a publishing sequence.")


def tab_search() -> None:
    st.caption("Keyword search uses the official API (100 quota units per search). Results are not a complete list of videos and are not a reliable ranking of audience demand.")
    c1, c2, c3 = st.columns([3, 1, 1])
    q = c1.text_input("Keyword or topic")
    order = c2.selectbox("Order", ["relevance", "viewCount", "date"])
    n = c3.slider("Results", 5, 50, 20)
    use_sample = c3.checkbox("Sample data", value=not youtube_key(), key="search_sample")
    if st.button("Search"):
        try:
            client = SampleClient() if use_sample else YouTubeClient(youtube_key())
            vids = client.search_videos(q, n, order)
            df = pd.DataFrame(vids)
            if df.empty:
                st.info("No videos found.")
            else:
                df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
                df["duration"] = df["duration_seconds"].apply(format_duration)
                st.session_state["search_results"] = df[["title", "channel_title", "url", "published_at", "views", "duration", "category"]]
        except (YouTubeAPIError, ValueError) as exc:
            st.error(f"{exc}  A SAMPLE-data option is available above.")
    res = st.session_state.get("search_results")
    if res is not None:
        show_df(res)
        csv_button("Download search results (CSV)", res, "search_results.csv", "dl_search")


def tab_export(r: dict) -> None:
    st.markdown("#### Sources")
    for s in r["sources"]:
        st.markdown(f"- {s['type']}: **{s['item']}** - {s['detail']}")
    with st.expander("Limitations and methodology notes", expanded=True):
        for lim in r["limitations"]:
            st.markdown(f"- {lim}")
    st.markdown("#### Downloads")
    st.caption("Files are built once from the stored results (no API or AI calls are repeated). No API keys are ever included.")
    version = r["meta"]["version"]
    if st.button("Build / refresh export files", type="primary"):
        with st.spinner("Building PDF, Excel, Word and JSON..."):
            st.session_state["exports"] = {"pdf": build_pdf(r), "xlsx": build_workbook(r), "docx": build_docx(r), "json": results_to_json_bytes(r)}
            st.session_state["exports_version"] = version
    ex = st.session_state.get("exports")
    if ex and st.session_state.get("exports_version") != version:
        st.warning("Results changed since these files were built (e.g. calendar or topics). Click the button above to refresh them.")
    if ex:
        c = st.columns(4)
        c[0].download_button("PDF report", ex["pdf"], "youtube_channel_report.pdf", "application/pdf", key="dl_pdf")
        c[1].download_button("Excel workbook", ex["xlsx"], "youtube_channel_analysis.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
        c[2].download_button("Word report", ex["docx"], "youtube_channel_report.docx",
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_docx")
        c[3].download_button("JSON", ex["json"], "youtube_channel_analysis.json", "application/json", key="dl_json")
    st.markdown("**CSV exports**")
    c = st.columns(3)
    with c[0]:
        csv_button("Video data (CSV)", r["videos"][["title", "url", "published_at", "views", "likes", "comments", "views_per_day", "relative_to_median", "format", "topic"]], "videos.csv", "dl_csv_v")
    with c[1]:
        csv_button("Competitor comparison (CSV)", r["competitors"], "competitors.csv", "dl_csv_c")
    with c[2]:
        csv_button("Content gaps (CSV)", r["gaps"], "content_gaps.csv", "dl_csv_g")


# --------------------------------------------------------------------------- main
def main() -> None:
    form = sidebar()
    if form:
        run_from_form(form)
    st.title("AI YouTube Channel Analyzer")
    r = st.session_state.get("results")
    if r is None:
        st.markdown("Analyse a channel's **public** performance, compare competitors, find content gaps and plan what to publish next.")
        st.info("Enter a channel in the sidebar and press **Run analysis**. With no API key, tick the SAMPLE-data option to explore the app with clearly-labelled illustrative data.")
        st.markdown("**What public data can't tell you:** watch time, impressions, click-through rate, retention, traffic sources and revenue are private to channel owners and are never shown or estimated here.")
        with st.expander("Keyword search", expanded=False):
            tab_search()
        return
    names = ["Overview", "Video Performance", "Title Analysis", "Topics", "Competitors", "Content Gaps", "AI Content Plan", "Search", "Sources and Export"]
    tabs = st.tabs(names)
    for tab, fn in zip(tabs, (tab_overview, tab_videos, tab_titles, tab_topics, tab_competitors, tab_gaps, tab_plan, None, tab_export)):
        with tab:
            if fn is None:
                tab_search()
            else:
                fn(r)


main()
