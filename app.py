"""AI YouTube Channel Analyzer - Streamlit app (public YouTube data, automatic competitor discovery, growth roadmap)."""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src import content_gap_analysis as gaps_a
from src import content_strategy as strat
from src import audience_profile as ap
from src import copyright_risk as crisk
from src import editing_advice as ea
from src import media_analysis as ma
from src import metadata_generator as mg
from src import monetization as mon
from src import shorts_predictor as sp
from src import trending as tr
from src import upload_planner as up
from src import roadmap as rm
from src import title_analysis as ta
from src.ai_provider import AIProvider, AIProviderError
from src.config import YOUTUBE_KEY_NAMES, is_truthy, read_setting, resolve_data_mode
from src.pipeline import apply_manual_topics, recompute_topics, refresh_monetization, refresh_roadmap, run_analysis
from src.sample_data import SampleClient
from src.topic_analysis import parse_topic_mapping
from src.youtube_api import (InvalidAPIKeyError, MissingAPIKeyError, QuotaExceededError, YouTubeAPIError,
                             YouTubeClient, estimate_quota)
from utils.data_processing import df_to_csv_bytes, format_number, results_to_json_bytes, top_terms
from utils.export_excel import build_workbook
from utils.export_pdf import build_pdf
from utils.export_word import build_docx
from utils.theme import COLORWAY, CSS, PALETTE, card, hero, kpi_card, section

st.set_page_config(page_title="AI YouTube Channel Analyzer", page_icon="📊", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- configuration (no user key prompts)
def _secrets():
    try:
        return st.secrets
    except Exception:  # noqa: BLE001 - no secrets file
        return None


def youtube_key() -> str:
    for name in YOUTUBE_KEY_NAMES:
        v = read_setting(name, _secrets())
        if v:
            return v
    return ""


def data_mode() -> str:
    force = is_truthy(read_setting("USE_SAMPLE_DATA", _secrets()))
    return resolve_data_mode(youtube_key(), force)


def ai_provider() -> AIProvider | None:
    cfg = {k: read_setting(k, _secrets()) for k in ("AI_PROVIDER", "AI_API_KEY", "AI_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    try:
        prov = AIProvider.from_config(cfg)
    except AIProviderError:
        return None
    return prov if prov.is_configured else None


# --------------------------------------------------------------------------- helpers
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
    for k in ("results", "exports", "exports_version", "tpl_text", "plan_progress"):
        st.session_state.pop(k, None)


def cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for i, (col, (label, value, sub)) in enumerate(zip(cols, items)):
        col.markdown(kpi_card(label, value, sub, i), unsafe_allow_html=True)


def style(fig, title: str | None = None):
    fig.update_layout(template="plotly_white", colorway=COLORWAY, title=title or fig.layout.title.text,
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(color=PALETTE["navy"]))
    return fig


# --------------------------------------------------------------------------- sidebar
def sidebar() -> dict | None:
    st.sidebar.markdown("## 📊 Channel Analyzer")
    mode = data_mode()
    if mode == "live":
        st.sidebar.markdown("🟢 **Live YouTube data**")
    else:
        st.sidebar.markdown("🟠 **Demo mode** (illustrative data)")
        st.sidebar.caption("No YouTube API key is configured on the server, so the app shows clearly-labelled sample data. The app owner adds the key in Streamlit secrets.")
    prov = ai_provider()
    st.sidebar.caption("✨ AI writing assistant: " + (f"on ({prov.provider})" if prov else "off - everything else works without it"))
    with st.sidebar.form("analysis_form"):
        channel = st.text_input("Your channel", placeholder="https://www.youtube.com/@yourchannel", help="Channel URL, @handle or channel ID")
        sample_size = st.slider("Videos analysed per channel", 10, 200, 50, step=5)
        n_comp = st.slider("Competitors to discover automatically", 10, 20, 10)
        use_dates = st.checkbox("Limit to a date range")
        dr = st.date_input("Date range", value=(date.today() - timedelta(days=365), date.today()), disabled=not use_dates)
        submitted = st.form_submit_button("🚀 Analyse my channel")
    if mode == "live":
        st.sidebar.caption(f"Approx. YouTube quota for this run: {estimate_quota(sample_size, 1) + 400 + estimate_quota(min(sample_size, 40), n_comp)} of 10,000 daily units.")
    st.sidebar.button("Start a new analysis", on_click=reset_analysis)
    if not submitted:
        return None
    date_from = date_to = None
    if use_dates and isinstance(dr, (tuple, list)) and len(dr) == 2:
        date_from = datetime.combine(dr[0], time.min, tzinfo=timezone.utc)
        date_to = datetime.combine(dr[1], time.max, tzinfo=timezone.utc)
    return {"channel": channel, "sample_size": sample_size, "n_comp": n_comp, "date_from": date_from, "date_to": date_to, "mode": mode}


def run_from_form(form: dict) -> None:
    reset_analysis()  # a new analysis never mixes with earlier results
    try:
        client = SampleClient() if form["mode"] == "sample" else YouTubeClient(youtube_key())
        bar = st.progress(0.0, text="Starting...")
        st.session_state["results"] = run_analysis(client, form["channel"], [], form["sample_size"], form["date_from"], form["date_to"],
                                                    auto_competitors=form["n_comp"], progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        bar.empty()
    except ValueError as exc:
        st.error(f"Input problem: {exc}")
    except (MissingAPIKeyError, InvalidAPIKeyError) as exc:
        st.error(f"The server's YouTube API key is missing or was rejected ({exc}). Please contact the app owner.")
    except QuotaExceededError as exc:
        st.error(f"{exc}")
    except YouTubeAPIError as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- tabs
def tab_overview(r: dict) -> None:
    life, samp, freq = r["summary"]["lifetime"], r["summary"]["sample"], r["summary"]["upload_frequency"]
    st.markdown(section(life["channel_name"] or "Channel"), unsafe_allow_html=True)
    if str(r["meta"]["mode"]).startswith("SAMPLE"):
        st.warning("DEMO / SAMPLE DATA: everything below is illustrative and is not real YouTube data.")
    for n in r["notes"]:
        if "SAMPLE DATA" not in n and not n.startswith("Competitors were chosen"):
            st.info(n)
    for w in r["warnings"]:
        st.warning(w)
    st.caption("Lifetime channel statistics (whole channel, public)")
    cards([("Subscribers", "Hidden" if life["subscribers_hidden"] else format_number(life["subscribers"]), "public count"),
           ("Total views", format_number(life["total_views"]), "lifetime"),
           ("Total videos", format_number(life["total_videos"]), "lifetime"),
           ("Avg views / video", format_number(life["lifetime_avg_views_per_video"]), "lifetime total / videos")])
    st.caption(f"Selected sample ({samp.get('videos_in_sample', 0)} videos)")
    if samp.get("videos_in_sample", 0) == 0:
        st.info("No public videos matched the selected sample/date range.")
        return
    cards([("Median views", format_number(samp["median_views"]), "sample"),
           ("Mean views", format_number(samp["mean_views"]), "pulled up by outliers"),
           ("Uploads / 30 days", "n/a" if freq["uploads_per_30_days"] is None else f"{freq['uploads_per_30_days']:.1f}", "from sample dates"),
           ("Short-form share", f"{samp['short_form_share']:.0%}", "videos of 3 min or less")])
    df = r["videos"]
    fig = px.scatter(df.sort_values("published_at"), x="published_at", y="views", hover_data=["title"], color="views", size="views", size_max=16,
                     color_continuous_scale=["#A29BFE", "#6C5CE7", "#0984E3", "#00B894"])
    fig.add_hline(y=samp["median_views"], line_dash="dash", line_color=PALETTE["pink"], annotation_text="sample median")
    st.plotly_chart(style(fig, "Public views by upload date (selected sample)"), width="stretch")
    pp = r["publishing_patterns"]
    if len(pp["weekday"]):
        a, b = st.columns(2)
        a.plotly_chart(style(px.bar(pp["weekday"], x="weekday", y="videos", color_discrete_sequence=[PALETTE["teal"]]), "Uploads by weekday (UTC)"), width="stretch")
        b.plotly_chart(style(px.bar(pp["weekday"], x="weekday", y="median_views", color_discrete_sequence=[PALETTE["pink"]]), "Median views by upload weekday (small samples are noisy)"), width="stretch")
    with st.expander("Channel description"):
        st.write(life["description"] or "No description.")
    csv_button("Download overview (CSV)", r["overview_table"], "channel_overview.csv", "dl_overview")


def _pct_of_bench(v, bench, higher):
    if v is None or bench is None or pd.isna(v) or pd.isna(bench) or not v or not bench:
        return None
    return (v / bench if higher else bench / v) * 100


def tab_roadmap(r: dict) -> None:
    road = r.get("roadmap") or {}
    if not road.get("available"):
        st.info("A roadmap needs at least one analysed video.")
        return
    k = road["kpis"]
    st.markdown(section("Channel health score and growth roadmap"), unsafe_allow_html=True)
    st.caption(f"Benchmarks: {road['benchmark_source']}. Targets are planning goals derived from peer benchmarks, not predictions or promises.")
    left, right = st.columns([1, 2])
    score = road["health_score"]
    if score is not None:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix": "/100"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": PALETTE["violet"]},
                                              "steps": [{"range": [0, 40], "color": "#FFE3E3"}, {"range": [40, 70], "color": "#FFF3C4"}, {"range": [70, 100], "color": "#D3F9E8"}]}))
        gauge.update_layout(height=250, margin=dict(l=30, r=40, t=50, b=0), title="Channel health score", font=dict(color=PALETTE["navy"], size=12))
        left.plotly_chart(gauge, width="stretch")
        left.caption("100 = at or above the peer median on every KPI")
    with right:
        mo = road["momentum"]
        behind = k[(k["attainment"].notna()) & (k["attainment"] < 1)]
        items = [("KPIs behind peers", f"{len(behind)} of {int(k['attainment'].notna().sum())}", "biggest levers first"),
                 ("Peers analysed", str(len(r["competitors"]) - 1 if len(r["competitors"]) else 0), "auto-discovered")]
        if mo.get("available") and pd.notna(mo.get("change_pct")):
            items.append(("Momentum (views/day)", f"{mo['change_pct']:+.0%}", "newer vs older half of sample"))
        cards(items)
        with st.expander("How the health score is calculated"):
            st.markdown("Score = weighted average of **attainment** across KPIs (your value / peer median, capped at 100%; inverted for 'lower is better' KPIs). "
                        "Weights: uploads 20%, per-day views 20%, views per 1,000 subscribers 20%, engagement 15%, breakout rate 15%, steadiness 10%. "
                        "Momentum is conservative because newer videos are still accumulating views.")
            if len(road["health_components"]):
                show_df(road["health_components"])

    st.markdown(section("KPIs: where you are, the benchmark, and your day 30/60/90 targets"), unsafe_allow_html=True)
    view = k[["metric", "you", "benchmark", "benchmark_source", "attainment", "status", "target_30", "target_60", "target_90"]].copy()
    for c in ("you", "benchmark", "target_30", "target_60", "target_90"):
        view[c] = view[c].astype(float).map(lambda v: float(f"{v:.3g}") if pd.notna(v) else v)
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={"metric": "Metric", "you": "You", "benchmark": "Benchmark", "benchmark_source": "Benchmark source", "status": "Status",
                                "target_30": "Day 30 target", "target_60": "Day 60 target", "target_90": "Day 90 target",
                                "attainment": st.column_config.ProgressColumn("Attainment", min_value=0.0, max_value=1.0, format="percent")})
    chart_rows = []
    for row in k.itertuples():
        for lab, col in (("Today", "you"), ("Day 30", "target_30"), ("Day 60", "target_60"), ("Day 90", "target_90")):
            p = _pct_of_bench(getattr(row, col), row.benchmark, row.higher_is_better)
            if p is not None:
                chart_rows.append({"KPI": row.metric.split(" (")[0], "When": lab, "% of peer benchmark": p})
    if chart_rows:
        fig = px.bar(pd.DataFrame(chart_rows), x="KPI", y="% of peer benchmark", color="When", barmode="group",
                     color_discrete_sequence=[PALETTE["violet"], PALETTE["teal"], PALETTE["blue"], PALETTE["pink"]])
        fig.add_hline(y=100, line_dash="dash", line_color=PALETTE["navy"], annotation_text="peer median")
        st.plotly_chart(style(fig, "Progress path toward peer benchmark (100% = peer median)"), width="stretch")

    st.markdown(section("Priority actions (largest weighted gap first)"), unsafe_allow_html=True)
    colors = [PALETTE["coral"], PALETTE["violet"], PALETTE["blue"], PALETTE["teal"], PALETTE["pink"], PALETTE["amber"]]
    for i, a in enumerate(road["actions"].itertuples()):
        steps = "".join(f"<li>{s}</li>" for s in a.how.split(" | "))
        st.markdown(card(f"{a.priority_rank}. {a.action}", f"<b>{a.kpi}</b><br>{a.why}<ul>{steps}</ul><i>Basis: {a.evidence_type}</i>",
                         colors[i % len(colors)], f"#{a.priority_rank}"), unsafe_allow_html=True)

    st.markdown(section("Day-by-day plan for the next 30 days"), unsafe_allow_html=True)
    plan = road["daily_plan"]
    phase = st.multiselect("Phase", list(plan["phase"].unique()), default=list(plan["phase"].unique()))
    edited = st.data_editor(plan[plan["phase"].isin(phase)], hide_index=True, width="stretch", key="plan_editor",
                            disabled=[c for c in plan.columns if c != "done"],
                            column_config={"done": st.column_config.CheckboxColumn("Done"), "task": st.column_config.TextColumn("Task", width="large")})
    done_pct = float(edited["done"].mean()) if len(edited) else 0.0
    st.progress(done_pct, text=f"Plan progress: {done_pct:.0%} of the selected days completed (tick the boxes as you go)")
    st.caption(f"Total planned effort: about {int(plan['est_minutes'].sum() / 60)} hours over 30 days. Uploads are placed on days that match the day-30 cadence goal.")
    csv_button("Download 30-day plan (CSV)", plan, "daily_plan.csv", "dl_plan")

    st.markdown(section("90-day milestones"), unsafe_allow_html=True)
    ph = road["phases"]
    if len(ph):
        pivot = ph.pivot_table(index="metric", columns="milestone", values="target", aggfunc="first")
        cur = ph.drop_duplicates("metric").set_index("metric")["current"].rename("Today")
        st.dataframe(pd.concat([cur, pivot], axis=1).reset_index(), hide_index=True, width="stretch")
        for m, f in ph.drop_duplicates("milestone")[["milestone", "focus"]].itertuples(index=False):
            st.markdown(f"- **{m}:** {f}")

    st.markdown(section("What-if calculator (planning arithmetic, not a forecast)"), unsafe_allow_html=True)
    uplift = st.slider("Assumed improvement in median views per video from better packaging", 0, 30, 10, step=5, format="%d%%") / 100
    sc = rm.build_scenarios(k, float(r["videos"]["views"].median()), uplift)
    if len(sc):
        fig = px.bar(sc, x="scenario", y="views_from_new_videos", color="scenario", color_discrete_sequence=[PALETTE["slate"], PALETTE["teal"], PALETTE["violet"]])
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, "Views from videos published in the next 90 days (videos x median views)"), width="stretch")
        st.dataframe(sc.drop(columns=["note"]), hide_index=True, width="stretch")
        st.caption(sc["note"].iloc[0])
    with st.expander("Notes and limits"):
        for n in road["notes"]:
            st.markdown(f"- {n}")
    if st.button("Save this assumption into the exports"):
        refresh_roadmap(r, uplift)
        st.rerun()


def _usd(v: float) -> str:
    return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def tab_income(r: dict) -> None:
    mo = r.get("monetization") or {}
    if not mo.get("available"):
        st.info("The income estimate needs at least one analysed video.")
        return
    a, b = mo["assumptions"], mo["baseline"]
    st.markdown(section("Income estimate and monetization roadmap"), unsafe_allow_html=True)
    st.warning("**This is an estimate, not your real income.** Revenue, watch hours and monetization status are private to the channel owner (YouTube Studio > Earn). "
               "Figures below come from public data plus the assumptions you can change here.")
    with st.expander("Assumptions (change these to match your channel)", expanded=not b["subs_known"]):
        with st.form("income_assumptions"):
            c1, c2 = st.columns(2)
            status = c1.selectbox("Monetization status", list(mon.STATUS_OPTIONS), index=list(mon.STATUS_OPTIONS).index(a["status"]),
                                  format_func=mon.STATUS_OPTIONS.get)
            niche = c2.selectbox("Niche (sets the assumed income per 1,000 views)", list(mon.NICHE_RPM), index=list(mon.NICHE_RPM).index(a["niche"]))
            c3, c4 = st.columns(2)
            view_pct = c3.slider("Average % of each long video watched", 20, 70, int(round(a["view_pct"] * 100)), step=5) / 100
            growth = c4.slider("Assumed monthly growth in views", 0, 10, int(round(a["monthly_growth"] * 100)), step=1, format="%d%%") / 100
            uplift = c3.slider("Packaging uplift in the third scenario", 0, 30, int(round(a["packaging_uplift"] * 100)), step=5, format="%d%%") / 100
            subs_override = c4.number_input("Your real subscriber count (0 = use public count)", min_value=0, value=int(a["subs_override"]), step=100,
                                            help="Use this if the creator hides subscribers or you want the exact number from YouTube Studio.")
            if st.form_submit_button("Update estimate"):
                refresh_monetization(r, {"status": status, "niche": niche, "view_pct": view_pct, "monthly_growth": growth,
                                         "packaging_uplift": uplift, "subs_override": int(subs_override)})
                st.rerun()

    t = mon.ads_threshold(pd.Timestamp.now(tz="UTC"))
    p, el, gaps = mo["potential_monthly_income"], mo["eligible_now"], mo["gaps"]
    st.markdown(section("1. How much does this channel earn now?"), unsafe_allow_html=True)
    if mo["earning_now"]:
        cards([("Estimated income now / month", f"{_usd(p['low'])} - {_usd(p['high'])}", f"mid {_usd(p['mid'])} (estimate)"),
               ("Subscribers", format_number(b["subs"]), "public count" if not a["subs_override"] else "entered by you"),
               ("Est. views per month", format_number(b["long_views_month"] + b["short_views_month"]), "uploads x median views"),
               ("Assumed income per 1,000 views", f"${mo['rpm']['long'][0]:g} - ${mo['rpm']['long'][2]:g}", a["niche"])])
    else:
        why = "estimated to be below the ads thresholds" if mo["status"] == "auto" else "you said it is not monetized yet"
        cards([("Estimated income now", "$0", why),
               ("Subscribers", format_number(b["subs"]), f"need {t['subs']:,}"),
               ("Est. watch hours (12 mo)", format_number(b["hours_12m"]), f"need {t['hours']:,}"),
               ("Income once monetized", f"{_usd(p['low'])} - {_usd(p['high'])}", f"mid {_usd(p['mid'])} / month at today's pace")])
    if mo["status"] == "auto":
        st.caption("Monetization status cannot be read from public data. 'Estimated' means the channel appears to meet or miss the thresholds based on the assumptions above; "
                   "choose 'Already monetized' or 'Not monetized yet' to override.")

    st.markdown(section("2. How long until it can earn? (monetization status)"), unsafe_allow_html=True)
    if mo["earning_now"] or el["ads"]:
        st.success("Treated as already monetized (your choice)." if mo["status"] == "monetized" else "By this estimate the channel already meets the ads-tier thresholds.")
    else:
        miss = []
        if gaps["subs_gap"]:
            miss.append(f"**{gaps['subs_gap']:,.0f}** more subscribers")
        miss.append(f"either **{gaps['hours_gap']:,.0f}** more public long-form watch hours **or** **{gaps['shorts_gap'] / 1e6:,.1f}M** more Shorts views (90 days)")
        st.info("To reach the ads tier you still need " + " and ".join(miss) + f". Rule used today: {t['subs']:,} subs + {t['hours']:,} hours or {t['shorts'] / 1e6:.0f}M Shorts views.")
    if mon.ads_threshold(pd.Timestamp.now(tz="UTC")) is mon.ADS_BEFORE:
        st.caption(f"Heads-up: from 2027-02-01 new applicants need {mon.ADS_AFTER['hours']:,} watch hours or {mon.ADS_AFTER['shorts'] / 1e6:.0f}M Shorts views. The timeline applies the rule in force in each month.")
    el_tbl = mo["eligibility"].copy()
    el_tbl.columns = [c.replace("_", " ").capitalize() for c in el_tbl.columns]
    st.dataframe(el_tbl, hide_index=True, width="stretch")
    st.caption("'In months' counts from today; first income allows about a month for YouTube's review. 'None' means not reached within 24 months.")

    proj = mo["projection"]
    fig = px.line(proj, x="date", y="subscribers", color="scenario", color_discrete_sequence=[PALETTE["slate"], PALETTE["teal"], PALETTE["violet"]])
    fig.add_hline(y=t["subs"], line_dash="dash", line_color=PALETTE["navy"], annotation_text=f"{t['subs']:,} subscribers")
    st.plotly_chart(style(fig, "Estimated subscribers by scenario"), width="stretch")
    fig = px.line(proj, x="date", y="watch_hours_12m", color="scenario", color_discrete_sequence=[PALETTE["slate"], PALETTE["teal"], PALETTE["violet"]])
    fig.add_hline(y=mon.ADS_BEFORE["hours"], line_dash="dash", line_color=PALETTE["navy"], annotation_text=f"{mon.ADS_BEFORE['hours']:,} h (until 2027-01-31)")
    fig.add_hline(y=mon.ADS_AFTER["hours"], line_dash="dot", line_color=PALETTE["coral"], annotation_text=f"{mon.ADS_AFTER['hours']:,} h (from 2027-02-01)")
    st.plotly_chart(style(fig, "Estimated rolling 12-month public watch hours by scenario"), width="stretch")

    st.markdown(section("3. Income roadmap"), unsafe_allow_html=True)
    st.markdown("**Milestones** (middle scenario: you reach the roadmap's upload targets):")
    steps = mo["steps"].copy()
    steps.columns = [c.replace("_", " ").capitalize() for c in steps.columns]
    st.dataframe(steps, hide_index=True, width="stretch", column_config={"Target": st.column_config.TextColumn(width="large")})
    inc = proj.copy()
    fig = go.Figure()
    for i, (name, g) in enumerate(inc.groupby("scenario", sort=False)):
        col = [PALETTE["slate"], PALETTE["teal"], PALETTE["violet"]][i % 3]
        fig.add_trace(go.Scatter(x=g["date"], y=g["income_mid"], name=name, mode="lines", line=dict(color=col)))
    mid = inc[inc["scenario"] == inc["scenario"].iloc[len(inc) // 2]]
    fig.add_trace(go.Scatter(x=list(mid["date"]) + list(mid["date"])[::-1], y=list(mid["income_high"]) + list(mid["income_low"])[::-1], fill="toself",
                             fillcolor="rgba(20,184,166,0.15)", line=dict(width=0), name="Low-high range (middle scenario)", hoverinfo="skip"))
    fig.update_yaxes(title="USD per month")
    st.plotly_chart(style(fig, "Estimated monthly income by scenario (mid RPM; band = low-high RPM)"), width="stretch")
    st.markdown("**What you would need to reach the ads tier sooner:**")
    rp = mo["reverse_plan"].copy()
    rp.columns = [c.replace("_", " ").capitalize() for c in rp.columns]
    st.dataframe(rp, hide_index=True, width="stretch")
    st.caption("Needs are simple arithmetic: missing hours spread over the months, converted to long-form views at your average duration and assumed % watched, then to videos at your median views.")
    csv_button("Download income projection (CSV)", proj, "income_projection.csv", "dl_income")
    with st.expander("Method, assumptions and limits"):
        for n in mo["notes"]:
            st.markdown(f"- {n}")


def get_client():
    return SampleClient() if data_mode() == "sample" else YouTubeClient(youtube_key())


CATEGORIES = {"All categories": None, "Gaming": "20", "Education": "27", "How-to & Style": "26", "Entertainment": "24", "Science & Technology": "28",
              "People & Blogs": "22", "Music": "10", "Sports": "17", "Comedy": "23", "News & Politics": "25", "Travel & Events": "19"}


def tab_trending(r: dict) -> None:
    st.markdown(section("Trending and high-potential videos"), unsafe_allow_html=True)
    st.info("**Sources:** YouTube's official API only. TikTok, Instagram, X and other platforms have no public trending API and forbid scraping, so this app does not scrape them. "
            "For those platforms, import a CSV from your own analytics export or a tool you are licensed to use. "
            "Study topics, formats and hooks; do not copy titles, thumbnails or footage (that creates copyright and reused-content risk).")
    subs = r["channel"].get("subscribers") or None
    mode = st.radio("What do you want to find?", ["Trending now (YouTube chart)", "High-potential in my niche", "Specific creators", "Other platforms (CSV import)"], horizontal=True)
    kw_default = ", ".join(k for k, _ in top_terms(r["videos"]["title"], 3))
    res, notes = None, []
    try:
        if mode.startswith("Trending"):
            c1, c2 = st.columns(2)
            region = c1.text_input("Region code", "US", max_chars=2, help="Two-letter country code, e.g. US, GB, BD, IN")
            cat = c2.selectbox("Category", list(CATEGORIES))
            st.caption("About 3 quota units.")
            if st.button("Find trending videos", type="primary"):
                with st.spinner("Asking YouTube..."):
                    res = tr.trending(get_client(), region or "US", CATEGORIES[cat], subs)
        elif mode.startswith("High-potential"):
            kws = st.text_input("Your niche keywords (up to 3, comma separated)", kw_default)
            c1, c2, c3 = st.columns(3)
            days = c1.slider("Published in the last (days)", 7, 90, 30)
            region = c2.text_input("Region code (optional)", "", max_chars=2)
            short_only = c3.checkbox("Short videos only")
            st.caption("About 100 quota units per keyword. Results from creators near your size are flagged because they are the most realistic to learn from.")
            if st.button("Find high-potential videos", type="primary"):
                with st.spinner("Searching YouTube..."):
                    res = tr.niche_outliers(get_client(), [k.strip() for k in kws.split(",")], days, region or None, subs, short_only)
        elif mode.startswith("Specific"):
            links = st.text_area("Creator channel links, @handles or IDs (one per line, up to 8)", height=110)
            st.caption("About 3-4 quota units per creator. 'Outlier' means views far above that creator's own typical video.")
            if st.button("Analyse these creators", type="primary") and links.strip():
                with st.spinner("Reading public uploads..."):
                    res, notes = tr.creator_outliers(get_client(), links.splitlines())
        else:
            st.download_button("Download CSV template", tr.TEMPLATE_CSV, "other_platforms_template.csv", "text/csv")
            up_file = st.file_uploader("Your CSV (columns: title, views, posted_at; optional platform, creator, url, likes, comments, followers)", type=["csv"])
            if up_file is not None:
                try:
                    res, notes = tr.import_external(pd.read_csv(up_file), subs)
                except Exception as exc:  # noqa: BLE001
                    notes = [f"The CSV could not be read: {exc}"]
    except (QuotaExceededError, YouTubeAPIError, MissingAPIKeyError, InvalidAPIKeyError) as exc:
        st.error(str(exc))
    if res is not None:
        st.session_state["trend_result"] = res
    for n in notes:
        st.warning(n)
    res = st.session_state.get("trend_result")
    if res is not None and len(res):
        cards([("Videos scored", f"{len(res):,}", "ranked by potential score"), ("Top score", f"{res['potential_score'].max():.0f}", "0-100, relative to this list"),
               ("Breakouts", f"{res['signal'].str.startswith(('Breakout', 'Outlier')).sum():,}", "far above audience size")])
        cols = [c for c in res.columns if c not in ("video_id",)]
        show_df(res[cols].round(3))
        csv_button("Download results (CSV)", res, "trending_videos.csv", "stu_dl_trend")
        st.caption("Potential score = 40% views per day + 40% views relative to audience size (capped) + 20% engagement, ranked within this list. It highlights what to study; it does not predict your results.")
    elif res is not None:
        st.info("Nothing found. Try other keywords, a wider date range or another region.")


def _studio_inputs(r: dict) -> dict:
    st.markdown(section("Your new video"), unsafe_allow_html=True)
    st.caption("Fill this once; every feature below uses it together with your channel's public results. Uploading the file lets the app measure loudness, pacing and size "
               "(analysed on the server, not stored). Nothing here is shared.")
    c1, c2 = st.columns(2)
    title = c1.text_input("Draft title (optional)", key="st_title")
    topic = c2.text_input("Topic / main keyword", key="st_topic", placeholder="e.g. home espresso")
    c3, c4 = st.columns(2)
    keywords = c3.text_input("Extra keywords (comma separated)", key="st_kw")
    points = c4.text_area("Key points of the video (one per line)", key="st_points", height=80)
    f = st.file_uploader("Optional: upload the video/audio file for measurements", type=[e.strip(".") for e in sorted(ma.ALLOWED_EXT)], key="st_file")
    if f is not None and st.button("Measure my file"):
        with st.spinner("Measuring audio, pacing and size (up to a few minutes for long files)..."):
            st.session_state["st_media"] = ma.analyze_upload(f.getvalue(), f.name)
    media = st.session_state.get("st_media")
    if media:
        if media.get("available"):
            st.success(f"Measured: {media['duration_seconds']:.0f}s, {media['width']}x{media['height']}"
                       + (f", {media['loudness_lufs']:.1f} LUFS" if media.get("loudness_lufs") is not None else "")
                       + (f", {media['cuts_per_minute']} cuts/min" if media.get("cuts_per_minute") is not None else ""))
        for n in media.get("notes", []):
            st.warning(n)
    guess_dur = float(media["duration_seconds"]) if media and media.get("available") else 30.0
    guess_short = bool(media and media.get("available") and media["duration_seconds"] <= 180 and media.get("vertical"))
    c5, c6 = st.columns(2)
    fmt = c5.radio("Format", ["short", "long"], index=0 if guess_short else 1, horizontal=True, format_func=lambda x: "YouTube Short" if x == "short" else "Long-form video", key="st_fmt")
    dur = c6.number_input("Duration (seconds)", 5, 14400, int(guess_dur), key=f"st_dur_{int(guess_dur)}")
    return {"title": title.strip(), "topic": topic.strip(), "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
            "points": [p.strip() for p in points.splitlines() if p.strip()], "fmt": fmt, "duration": float(dur), "media": media if media and media.get("available") else None}


def tab_studio(r: dict) -> None:
    prof = r.get("audience_profile") or ap.build_profile(r["videos"])
    if not prof.get("available"):
        st.info("Analyse a channel with at least a few videos to use the studio.")
        return
    v = _studio_inputs(r)
    st.caption(f"Channel profile: {prof['n_videos']} videos ({prof['n_short']} Shorts), reliability **{prof['reliability']}**. " + prof["note"])
    t1, t2, t3, t4, t5 = st.tabs(["🎞️ Editing & music", "✍️ Title, caption & hashtags", "📈 Shorts predictor", "🗓️ Upload planner", "🛡️ Copyright & reuse risk"])
    with t1:
        inferred, why = ea.infer_mood(r["videos"])
        mood = st.selectbox("Music / editing style", list(ea.MOODS), index=list(ea.MOODS).index(inferred))
        st.caption(f"Suggested: **{inferred}**. {why} Change it if you disagree.")
        has_voice = st.checkbox("The video has talking / voice-over", value=True)
        adv = ea.build_advice(v["fmt"], v["duration"], mood, prof, v["media"])
        show_df(adv)
        mp = ea.music_plan(v["fmt"], mood, has_voice)
        cards([("Music mood", mp["mood"], mp["genre"][:60]), ("Tempo", f"{mp['bpm'][0]}-{mp['bpm'][1]} BPM", f"cut about every {mp['cut_every_seconds']}s on the beat"),
               ("Level", "Under the voice", "12-18 dB below speech")])
        st.markdown(f"**Mix:** {mp['mix']}\n\n**Structure:** {mp['structure']}")
        st.markdown("**Where to get music you can use:**\n" + "\n".join(f"- {s}" for s in mp["sources"]))
        st.info("**Test it:** " + mp["ab_test"])
        st.caption("Public data cannot show which music a viewer prefers, so this is a standard production plan matched to your best-performing content style, not a measured preference.")
        csv_button("Download editing advice (CSV)", adv, "editing_advice.csv", "stu_dl_edit")
    with t2:
        c1, c2 = st.columns(2)
        goal = c1.selectbox("Main goal", list(strat.CTAS))
        audience = c2.text_input("Who is it for?", key="st_aud")
        other = st.text_input("Compare with (optional, for 'X vs Y' titles)", key="st_other")
        prov = ai_provider()
        use_ai = st.checkbox("Also ask the AI assistant for title ideas", disabled=prov is None, help=None if prov else "AI assistant is not configured")
        if st.button("Generate titles, caption and hashtags", type="primary"):
            try:
                st.session_state["st_meta"] = mg.generate(prof, v["topic"] or v["title"], v["keywords"], v["points"], v["fmt"], v["duration"], goal, audience, other, prov if use_ai else None)
            except ValueError as exc:
                st.warning(str(exc))
        m = st.session_state.get("st_meta")
        if m:
            show_df(m["titles"])
            st.markdown("**Caption / description**")
            st.code(m["description"], language=None)
            c3, c4 = st.columns(2)
            c3.markdown("**Hashtags**")
            c3.code(" ".join(m["hashtags"]), language=None)
            c4.markdown("**Tags (optional)**")
            c4.code(", ".join(m["tags"]), language=None)
            st.markdown("**Pinned comment idea**")
            st.code(m["pinned_comment"], language=None)
            st.caption(m["note"])
            csv_button("Download title options (CSV)", m["titles"], "title_options.csv", "stu_dl_titles")
    with t3:
        if v["fmt"] != "short":
            st.info("Switch the format above to **YouTube Short** to predict a Short.")
        topics = ["(none)"] + (list(r["topics"]["topic"]) if len(r["topics"]) else [])
        c1, c2 = st.columns(2)
        pdate = c1.date_input("Planned publish date", value=date.today() + timedelta(days=1), key="st_pdate")
        ptopic = c2.selectbox("Closest topic on your channel", topics)
        if st.button("Predict Short performance", type="primary"):
            if not (v["title"] or v["topic"]):
                st.warning("Enter a draft title or topic above first.")
            else:
                row = sp.feature_row(v["title"] or v["topic"], v["duration"], pd.Timestamp(pdate, tz="UTC"), "", "" if ptopic == "(none)" else ptopic)
                st.session_state["st_pred"] = sp.predict(r["videos"], row)
        pr = st.session_state.get("st_pred")
        if pr:
            if not pr["available"]:
                st.warning(pr["message"])
            else:
                cards([("Typical settled views", f"{format_number(pr['range_low'])} - {format_number(pr['range_high'])}", f"median {format_number(pr['expected_median'])}"),
                       ("Chance above your Shorts median", f"{pr['chance_above_channel_median'] * 100:.0f}%", f"your median {format_number(pr['baseline_median'])}"),
                       ("Confidence", pr["confidence"], f"{pr['n']} of your Shorts")])
                (st.success if pr["features_beat_baseline"] else st.info)(pr["message"])
                if pr["drivers"]:
                    show_df(pd.DataFrame(pr["drivers"]))
                for c in pr["caveats"]:
                    st.caption("• " + c)
    with t4:
        c1, c2, c3 = st.columns(3)
        tz = c1.number_input("Your time zone (UTC offset, hours)", -12.0, 14.0, 0.0, 0.5, help="Bangladesh = 6")
        horizon = c2.radio("Plan length", [7, 14, 30], index=1, horizontal=True, format_func=lambda d: f"{d} days")
        ppw = c3.slider("Posts per week", 1, 7, 3, key="pl_ppw")
        formats = st.multiselect("Formats", strat.FORMATS, default=["YouTube Short", "Long-form video"], key="pl_fmt")
        niche = st.text_input("Niche (optional)", key="pl_niche")
        slots = up.slot_table(r["videos"], tz)
        if slots["available"]:
            piv = slots["slots"].pivot(index="weekday", columns="hour_block", values="score").reindex(up.DAYS)
            fig = px.imshow(piv, color_continuous_scale="Teal", aspect="auto", labels=dict(color="Slot score"))
            st.plotly_chart(style(fig, f"Your past publish slots (1.0 = typical; reliability: {slots['reliability']}; time zone UTC{tz:+g})"), width="stretch")
            top = slots["slots"].head(3)
            st.caption("Best past slots: " + "; ".join(f"{x.weekday} {x.hour_block} ({x.evidence} evidence)" for x in top.itertuples()) +
                       ". These are when your better videos were published, not when viewers are online: confirm in YouTube Studio > Analytics > Audience.")
        if st.button("Build upload plan", type="primary"):
            st.session_state["st_plan"] = up.build_plan(r["videos"], r["gaps"], r["topics"], days=horizon, posts_per_week=ppw, formats=formats or None, niche=niche, tz_offset_hours=tz)
        plan = st.session_state.get("st_plan")
        if plan is not None and len(plan):
            show_df(plan)
            csv_button("Download upload plan (CSV)", plan, "upload_plan.csv", "stu_dl_plan")
            st.caption("'Typical views' = the 25th-75th percentile of your own settled videos in that format, scaled by the slot score. Topics come from your content gaps and are evidence-labelled.")
        with st.expander("After-upload checklist"):
            for c in up.CHECKLIST:
                st.markdown(f"- {c}")
    with t5:
        c1, c2 = st.columns(2)
        music = c1.selectbox("Music", list(crisk.MUSIC), index=len(crisk.MUSIC) - 1)
        clips = c2.selectbox("Other people's video clips", list(crisk.CLIPS))
        c3, c4 = st.columns(2)
        images = c3.selectbox("Images / graphics", list(crisk.IMAGES))
        voice = c4.selectbox("Voice and script", list(crisk.VOICE))
        c5, c6, c7 = st.columns(3)
        perm = c5.checkbox("I have written permission/licence for third-party footage")
        commentary = c6.checkbox("I add substantial original commentary or transformation", value=True)
        ai_real = c7.checkbox("Contains realistic AI-generated/altered content")
        ai_disc = st.checkbox("...and I disclose it in the upload settings", disabled=not ai_real)
        desc = st.text_area("Draft description (optional, for the wording scan)", key="cr_desc", height=80)
        if st.button("Check risk", type="primary"):
            st.session_state["st_risk"] = crisk.assess(v["title"] or v["topic"], desc, v["keywords"], music, clips, images, voice, perm, commentary, ai_real, ai_disc,
                                                       r["videos"], v["media"], v["duration"])
        rk = st.session_state.get("st_risk")
        if rk:
            (st.error if rk["level"] == "High" else st.warning if rk["level"] == "Medium" else st.success)(f"**{rk['level']} risk** (score {rk['score']}/100). {rk['summary']}")
            show_df(rk["findings"])
            st.caption(rk["disclaimer"])
            csv_button("Download risk findings (CSV)", rk["findings"], "risk_findings.csv", "stu_dl_risk")


def tab_videos(r: dict) -> None:
    df = r["videos"]
    if df.empty:
        st.info("No videos to analyse.")
        return
    with st.expander("Methodology (how each number is calculated)"):
        st.markdown("- **Views per day** = public views / days since publication (min 1 day). Older videos have had longer to accumulate views.\n"
                    "- **Relative to median** = views / median views of this sample.\n"
                    "- **Engagement rate** = (likes + comments) / views using whatever is public; blank when both are hidden.\n"
                    "- **Below typical** = 0.5x median or less *and* older than 30 days; newer videos are *too new to judge*.\n"
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
    a.plotly_chart(style(px.histogram(view, x="relative_to_median", nbins=20, color_discrete_sequence=[PALETTE["violet"]]), "Distribution: views relative to sample median"), width="stretch")
    b.plotly_chart(style(px.scatter(view, x="age_days", y="views", hover_data=["title"], color="format", log_y=True, color_discrete_sequence=COLORWAY), "Views vs video age (days)"), width="stretch")
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
    st.caption("Patterns are observational: they show what co-occurs with higher or lower views in this sample, not what causes it. Groups under 3 videos are flagged.")
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

    st.markdown(section("Title ideas for a selected video"), unsafe_allow_html=True)
    pick = st.selectbox("Choose a video", df["title"].tolist())
    row = df[df["title"] == pick].iloc[0]
    kws = df.attrs.get("title_keywords") or []
    ideas = ta.suggest_titles_rule_based(pick, kws)
    prov = ai_provider()
    if st.button("Generate AI alternatives", disabled=prov is None, help=None if prov else "AI assistant is not configured"):
        try:
            with st.spinner("Asking the AI provider..."):
                r["title_suggestions"][pick] = ta.suggest_titles_ai(prov, pick, row["description"])
        except (AIProviderError, ValueError) as exc:
            st.warning(f"AI suggestions unavailable ({exc}); showing rule-based starting points.")
    shown = r["title_suggestions"].get(pick) or ideas
    label = "AI-generated" if pick in r["title_suggestions"] else "Rule-based starting points"
    st.caption(f"{label}: recommendations, not guaranteed improvements. Check each still describes what the video delivers.")
    for i, idea in enumerate(shown, 1):
        st.markdown(f"**{i}. {idea['title']}**  \n{idea['improvement']}")

    st.markdown(section("Thumbnail statistics (optional)"), unsafe_allow_html=True)
    st.caption("Simple image statistics from public thumbnails (brightness, contrast, saturation, edge busyness). Text density is not measured and none of this says anything about click-through rate.")
    if st.button("Analyse up to 20 thumbnails"):
        def fetch(url: str) -> bytes:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            return resp.content
        with st.spinner("Downloading thumbnails..."):
            r["thumbnails"] = ta.analyze_thumbnails(df, fetch, limit=20)
        if r["thumbnails"].empty:
            st.info("No thumbnails could be analysed (demo data has none, or downloads failed).")
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
    a.plotly_chart(style(px.bar(t, x="topic", y="videos", color="topic", color_discrete_sequence=COLORWAY), "Videos per topic"), width="stretch")
    b.plotly_chart(style(px.bar(t, x="topic", y="median_views", color="topic", color_discrete_sequence=COLORWAY), "Median views per topic"), width="stretch")
    if len(r["topic_trend"]):
        st.markdown("**Recent vs older content** (topic share in the newer vs older half of the sample)")
        show_df(r["topic_trend"])
    with st.expander("Inspect and correct categories"):
        st.markdown("Define your own topics (one per line, `Topic: keyword, keyword`) or correct individual videos. The roadmap and content gaps update.")
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
        st.info("No similar channels could be found automatically for this channel (see warnings on the Overview tab).")
        return
    st.markdown(section("Automatically discovered competitors"), unsafe_allow_html=True)
    st.caption("The app searches recent videos on your own topics, collects the channels behind them and keeps the best topical and audience-size matches. "
               "This is a heuristic: review the list before drawing conclusions.")
    disc = r["discovered_competitors"]
    if len(disc):
        st.dataframe(disc[["channel", "subscribers", "query_hits", "size_similarity", "match_score", "why", "url"]], hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("Channel", display_text="open"),
                                    "match_score": st.column_config.ProgressColumn("Match score", min_value=0.0, max_value=1.0)})
    st.markdown(section("Side-by-side comparison"), unsafe_allow_html=True)
    st.caption("Raw views do not show which creator is better: channels differ in age, audience size, niche and history.")
    show_df(comp)
    csv_button("Download competitor comparison (CSV)", comp, "competitor_comparison.csv", "dl_comp")
    a, b = st.columns(2)
    a.plotly_chart(style(px.bar(comp, x="channel", y="median_views", color="role", color_discrete_sequence=[PALETTE["violet"], PALETTE["teal"]]), "Median views (sample)"), width="stretch")
    b.plotly_chart(style(px.bar(comp, x="channel", y="uploads_per_30_days", color="role", color_discrete_sequence=[PALETTE["violet"], PALETTE["teal"]]), "Uploads per 30 days"), width="stretch")
    for line in r["competitor_differences"][:12]:
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
        st.write("No content-gap ideas could be generated from this sample. Try a larger sample.")
        return
    with st.expander("How the priority score works"):
        st.markdown("Priority (0-100) = **30% relevance** + **30% demand evidence** (competitor videos' performance vs their own channel medians, and how many there are) "
                    "+ **20% differentiation** (how little you cover it) + **20% feasibility** (fit with the format you already make). "
                    "Confidence reflects how many videos and channels support the idea. There is no search-volume data in this score.")
    prov = ai_provider()
    if st.button("Refine top ideas with AI", disabled=prov is None, help=None if prov else "AI assistant is not configured"):
        with st.spinner("Asking the AI provider..."):
            r["gaps"] = gaps_a.refine_gaps_with_ai(prov, gaps)
            refresh_roadmap(r)
        st.rerun()
    show_df(gaps[["rank", "gap_type", "topic", "suggested_title", "format", "priority_score", "confidence"]])
    csv_button("Download content gaps (CSV)", gaps, "content_gaps.csv", "dl_gaps")
    fig = px.bar(gaps.head(10).iloc[::-1], x="priority_score", y="topic", orientation="h", color="confidence",
                 color_discrete_map={"High": PALETTE["teal"], "Medium": PALETTE["amber"], "Low": PALETTE["pink"]})
    st.plotly_chart(style(fig, "Top opportunities by priority score"), width="stretch")
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
    use_ai = st.checkbox("Refine wording with AI", value=False, disabled=prov is None, help=None if prov else "AI assistant is not configured")
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


def tab_export(r: dict) -> None:
    st.markdown(section("Sources"), unsafe_allow_html=True)
    for s in r["sources"]:
        st.markdown(f"- {s['type']}: **{s['item']}** - {s['detail']}")
    with st.expander("Limitations and methodology notes", expanded=True):
        for lim in r["limitations"]:
            st.markdown(f"- {lim}")
    st.markdown(section("Downloads"), unsafe_allow_html=True)
    st.caption("Files are built once from the stored results (no API or AI calls are repeated). No API keys are ever included.")
    version = r["meta"]["version"]
    if st.button("Build / refresh export files", type="primary"):
        with st.spinner("Building PDF, Excel, Word and JSON..."):
            st.session_state["exports"] = {"pdf": build_pdf(r), "xlsx": build_workbook(r), "docx": build_docx(r), "json": results_to_json_bytes(r)}
            st.session_state["exports_version"] = version
    ex = st.session_state.get("exports")
    if ex and st.session_state.get("exports_version") != version:
        st.warning("Results changed since these files were built (e.g. calendar, topics or roadmap). Click the button above to refresh them.")
    if ex:
        c = st.columns(4)
        c[0].download_button("PDF report", ex["pdf"], "youtube_channel_report.pdf", "application/pdf", key="dl_pdf")
        c[1].download_button("Excel workbook", ex["xlsx"], "youtube_channel_analysis.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
        c[2].download_button("Word report", ex["docx"], "youtube_channel_report.docx",
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_docx")
        c[3].download_button("JSON", ex["json"], "youtube_channel_analysis.json", "application/json", key="dl_json")
    st.markdown("**CSV exports**")
    c = st.columns(4)
    with c[0]:
        csv_button("Video data (CSV)", r["videos"][["title", "url", "published_at", "views", "likes", "comments", "views_per_day", "relative_to_median", "format", "topic"]], "videos.csv", "dl_csv_v")
    with c[1]:
        csv_button("Competitors (CSV)", r["competitors"], "competitors.csv", "dl_csv_c")
    with c[2]:
        csv_button("Content gaps (CSV)", r["gaps"], "content_gaps.csv", "dl_csv_g")
    with c[3]:
        csv_button("30-day plan (CSV)", (r.get("roadmap") or {}).get("daily_plan"), "daily_plan.csv", "dl_csv_p")


# --------------------------------------------------------------------------- main
def main() -> None:
    form = sidebar()
    if form:
        run_from_form(form)
    st.markdown(hero("AI YouTube Channel Analyzer", "Understand your public performance, benchmark against automatically-found peers, and follow a day-by-day growth roadmap."),
                unsafe_allow_html=True)
    r = st.session_state.get("results")
    if r is None:
        cards([("1  Enter your channel", "Paste a link", "URL, @handle or ID"), ("2  Auto benchmark", "10+ peers", "found by topic match"),
               ("3  Growth roadmap", "Day by day", "targets from peer KPIs"), ("4  Income", "Estimate + roadmap", "time to monetization"), ("5  Upload Studio", "Edit, titles, plan", "music, Shorts, copyright"), ("6  Export", "PDF / Excel", "Word, CSV, JSON")])
        st.info("Enter your channel in the sidebar and press **Analyse my channel**. Competitors are found for you automatically.")
        st.markdown("**What public data can't tell you:** watch time, impressions, click-through rate, retention, traffic sources and revenue are private to channel owners. The Income tab shows clearly-labelled *estimates* only, never actual revenue.")
        return
    names = ["Overview", "🚀 Growth Roadmap", "💰 Income & Monetization", "🔥 Trending & Outliers", "🎬 Upload Studio", "Video Performance", "Title Analysis", "Topics", "Competitors", "Content Gaps", "AI Content Plan", "Sources and Export"]
    tabs = st.tabs(names)
    for tab, fn in zip(tabs, (tab_overview, tab_roadmap, tab_income, tab_trending, tab_studio, tab_videos, tab_titles, tab_topics, tab_competitors, tab_gaps, tab_plan, tab_export)):
        with tab:
            fn(r)


main()
