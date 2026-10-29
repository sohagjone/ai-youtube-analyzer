"""AI YouTube Channel Analyzer - Streamlit app (public YouTube data, automatic competitor discovery, growth roadmap)."""
from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src import content_gap_analysis as gaps_a
from src import content_strategy as strat
from src import monetization as mon
from src import platforms as pf
from src import roadmap as rm
from src import growth_predictor as gpred
from src import studio as stu
from src import trends as trn
from src import upload_planner as upl
from src import video_probe as vprobe
from src import title_analysis as ta
from src.ai_provider import AIProvider, AIProviderError
from src.config import YOUTUBE_KEY_NAMES, is_truthy, read_setting, resolve_data_mode
from src.pipeline import apply_manual_topics, recompute_topics, refresh_monetization, refresh_roadmap, run_analysis
from src.sample_data import SampleClient
from src.topic_analysis import parse_topic_mapping
from src.youtube_api import (InvalidAPIKeyError, MissingAPIKeyError, QuotaExceededError, YouTubeAPIError,
                             YouTubeClient, estimate_quota)
from utils.data_processing import df_to_csv_bytes, format_number, results_to_json_bytes
from utils.export_excel import build_workbook
from utils.export_pdf import build_pdf
from utils.export_studio import build_studio_workbook
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
    for k in ("results", "exports", "exports_version", "tpl_text", "plan_progress", "studio", "trend_chart", "radar", "trend_terms", "viral"):
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


# --------------------------------------------------------------------------- Trend Radar
def get_client():
    return SampleClient() if data_mode() == "sample" else YouTubeClient(youtube_key())


PLATFORM_SECRETS = ("X_BEARER_TOKEN", "TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "META_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID")


def get_connectors():
    secrets = {k: read_setting(k, _secrets()) for k in PLATFORM_SECRETS}
    sample = data_mode() == "sample"
    return pf.build_connectors(None if sample else YouTubeClient(youtube_key()), secrets, sample)


def _default_niche(r: dict) -> str:
    detected = stu.detect_niche(r["videos"])
    return detected if detected in pf.NICHE_PRESETS else "football"


def tab_trends(r: dict) -> None:
    st.markdown(section("Trend Radar: what is going viral right now"), unsafe_allow_html=True)
    st.caption("Official platform APIs only - no scraping. Platforms the app owner has not connected are shown as 'not connected' below.")
    st.markdown("#### 1. Viral and trending videos in your niche, across platforms")
    keys = list(pf.NICHE_PRESETS)
    c1, c2, c3 = st.columns([1, 2, 1])
    niche = c1.selectbox("Niche", keys, index=keys.index(_default_niche(r)), format_func=lambda k: pf.NICHE_PRESETS[k]["label"])
    custom = c2.text_input("Or your own keywords (comma separated)", placeholder="e.g. premier league goals, messi skills")
    days = c3.slider("Last (days)", 1, 30, 7)
    connectors = get_connectors()
    with st.expander("Platform connections"):
        st.dataframe(pf.status_table(connectors), hide_index=True, width="stretch")
        st.caption("X recent search covers the last 7 days. TikTok's Research API is limited to approved researchers (30-day windows). Instagram hashtag search returns popular posts without view counts and allows 30 unique hashtags per 7 days. "
                   "Facebook offers no public video search. The app owner adds credentials in Streamlit secrets; users never enter keys.")
    with st.expander("TikTok without Research API approval"):
        st.markdown("TikTok only lets **approved research projects** query videos (the Research API); creators, advertisers and commercial users are not eligible, and ordinary developer keys get a 401 error. "
                    "Instead, open TikTok's own free **[Creative Center](https://ads.tiktok.com/business/creativecenter/)** (trending hashtags, songs and videos by country and industry), "
                    "then add the hashtags you see here so the title, tag and hashtag generator can use them.")
        manual = st.text_input("Trending TikTok hashtags you found (comma separated)", key="tiktok_manual", placeholder="#footballskills, #goalchallenge")
    available = [c.name for c in connectors if c.configured()]
    chosen = st.multiselect("Platforms to search", available, default=available)
    if st.button("🔍 Find viral videos in my niche"):
        cfg = pf.niche_config(niche, custom)
        with st.spinner("Searching the connected platforms..."):
            df, warn = pf.discover(connectors, cfg, days, chosen)
        st.session_state["viral"] = {"df": df, "warnings": warn, "label": cfg["label"], "days": days, "keywords": cfg["keywords"]}
    V = st.session_state.get("viral")
    viral_terms: list[str] = []
    if V:
        for w in V["warnings"]:
            st.warning(w)
        df = V["df"]
        if df.empty:
            st.info("No videos found. Connect more platforms, widen the time window or change the keywords.")
        else:
            if data_mode() == "sample":
                st.warning("DEMO DATA: these cross-platform results are illustrative, not real.")
            summ = pf.platform_summary(df)
            cards([("Videos found", str(len(df)), f"{V['label']}, last {V['days']} days"), ("Trending now", str(int((df['status'] == 'Trending now').sum())), "posted within 48h and top-scoring"),
                   ("Platforms", str(df["platform"].nunique()), ", ".join(sorted(df["platform"].unique()))), ("Top score", f"{df['viral_score'].max():.0f}", "percentile within its platform")])
            pick = st.multiselect("Show platforms", sorted(df["platform"].unique()), default=sorted(df["platform"].unique()), key="viral_filter")
            view = df[df["platform"].isin(pick)]
            cols = ["platform", "title", "url", "author", "status", "viral_score", "views", "likes", "comments", "shares", "age_hours", "duration_seconds"]
            st.dataframe(view[cols], hide_index=True, width="stretch", height=420,
                         column_config={"url": st.column_config.LinkColumn("Link", display_text="open"),
                                        "viral_score": st.column_config.ProgressColumn("Viral score", min_value=0, max_value=100, format="%.0f"),
                                        "age_hours": st.column_config.NumberColumn("Age (h)", format="%.0f")})
            st.caption("Viral score = percentile of how fast a video gains views/engagement compared with the other results from the SAME platform, so platforms are not directly comparable. "
                       "Engagement = likes + 2x comments + 3x shares; Instagram returns no view counts. Metrics are as of the moment of the search.")
            csv_button("Download viral videos (CSV)", view[cols], "viral_videos.csv", "dl_viral")
            tags = pf.top_hashtags(df)
            if len(tags):
                st.plotly_chart(style(px.bar(tags, x="hashtag", y="viral_posts", color_discrete_sequence=[PALETTE["pink"]]), "Hashtags most used by the high-scoring videos"), width="stretch")
            for line in pf.what_they_share(df):
                st.markdown(f"- {line}")
            viral_terms = tags["hashtag"].tolist()[:8] if len(tags) else []
            st.dataframe(summ, hide_index=True, width="stretch")

    st.markdown("#### 2. YouTube's official most-popular chart")
    c1, c2, c3 = st.columns([1, 2, 1])
    region = c1.selectbox("Region", ["US", "GB", "IN", "BD", "PK", "CA", "AU", "DE", "FR", "BR", "JP", "KR", "ID", "NG", "AE", "TR"], index=0)
    cat = c2.selectbox("Category", list(trn.TREND_CATEGORIES))
    n = c3.slider("Videos", 10, 50, 25, step=5)
    if st.button("Show trending chart"):
        try:
            with st.spinner("Reading YouTube's chart..."):
                st.session_state["trend_chart"] = trn.trending_chart(get_client(), region, trn.TREND_CATEGORIES[cat], n)
            if st.session_state["trend_chart"].empty:
                st.info("YouTube returned no chart for that region/category combination. Try 'All categories'.")
        except YouTubeAPIError as exc:
            st.error(f"{exc} (some categories have no chart in some regions)")
    chart = st.session_state.get("trend_chart")
    if chart is not None and len(chart):
        cols = ["title", "url", "views", "views_per_hour", "age_days", "duration", "format", "pattern"]
        show_df(chart[cols], height=360)
        csv_button("Download trending chart (CSV)", chart[cols], "trending_chart.csv", "dl_trend_chart")
        L = trn.learnings(chart.assign(status="x"), 15)
        if L.get("available"):
            st.markdown(f"**What the fastest {L['n']} share:** {L['short_form_share']:.0%} short-form, median length {L['median_duration']}, common title patterns "
                        f"{', '.join(L['common_patterns'])}; common words: {', '.join(L['common_terms'])}. *{L['note']}*")
    st.markdown("#### 3. High-potential videos from creators you choose (YouTube)")
    st.caption("Paste up to 10 YouTube creators (channel links or @handles), such as players or channels in your niche. A video is flagged when its views per day are well above that creator's own usual pace.")
    text = st.text_area("Creators (one per line)", height=110, placeholder="@creator1\nhttps://www.youtube.com/@creator2", key="radar_text")
    window = st.slider("Look at videos published in the last (days)", 7, 60, 30)
    if st.button("Find high-potential videos"):
        bar = st.progress(0.0, text="Starting...")
        radar, warn = trn.creator_radar(get_client(), text.splitlines(), 30, window, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        bar.empty()
        st.session_state["radar"] = {"table": radar, "warnings": warn}
    rad = st.session_state.get("radar")
    if rad:
        for w in rad["warnings"]:
            st.warning(w)
        tbl = rad["table"]
        if tbl.empty:
            st.info("No recent videos found for those creators in that window.")
        else:
            hp = trn.high_potential(tbl)
            cards([("Videos checked", str(len(tbl)), f"last {window} days"), ("High potential", str(len(hp)), "above usual pace"), ("Breakouts", str(int((tbl['status'] == 'Breakout').sum())), f"{trn.BREAKOUT}x+ usual pace")])
            cols = ["creator", "title", "url", "status", "pace_ratio", "views", "views_per_day", "age_days", "format", "duration", "why"]
            show_df(tbl[cols], height=380)
            csv_button("Download creator radar (CSV)", tbl[cols], "creator_radar.csv", "dl_radar")
            L = trn.learnings(hp if len(hp) else tbl, 10)
            if L.get("available"):
                st.markdown(f"**What the fastest videos share:** {L['short_form_share']:.0%} short-form; median length {L['median_duration']}; patterns: {', '.join(L['common_patterns'])}; words: {', '.join(L['common_terms'])}. *{L['note']}*")
    manual_tags = [re.sub(r"[^0-9A-Za-z]", "", t).lower() for t in st.session_state.get("tiktok_manual", "").split(",")]
    terms = trn.trend_terms(chart if chart is not None else None, (rad or {}).get("table") if rad else None) + viral_terms + [t for t in manual_tags if t]
    st.session_state["trend_terms"] = list(dict.fromkeys(terms))
    if st.session_state["trend_terms"]:
        st.caption("Trending words found here are passed to the Video Studio's title and hashtag generator: " + ", ".join(st.session_state["trend_terms"][:15]))


# --------------------------------------------------------------------------- Video Studio
def _fmt_probe_cards(p: dict) -> None:
    ld, sl, cu = p.get("loudness") or {}, p.get("silence") or {}, p.get("cuts") or {}
    cards([("Length", f"{p['duration_seconds']:.0f}s", f"{p.get('width')}x{p.get('height')} {p.get('orientation', '')}"),
           ("Loudness", f"{ld['integrated_lufs']:.1f} LUFS" if ld.get("integrated_lufs") is not None else "n/a", "YouTube reference about -14"),
           ("Visual cuts", f"{cu.get('cuts_per_minute', 0):.1f}/min", f"avg shot {cu.get('avg_shot_seconds', 0):.1f}s"),
           ("Silent gaps", str(sl.get("silent_gaps", 0)), f"{sl.get('silent_seconds', 0):.1f}s total")])


def tab_studio(r: dict) -> None:
    st.markdown(section("Video Studio: upload a video and the AI does the rest"), unsafe_allow_html=True)
    prov = ai_provider()
    if prov:
        st.success("AI video understanding is ON: upload a video and the topic, titles, description and tags are worked out automatically.")
    else:
        st.warning("AI video understanding is OFF (the app owner has not added an AI key). Everything measurable still works, but the app cannot see what your video is about, so please type the topic.")
    st.info("Everything is tuned to **your channel's public performance** (views, views/day, likes and comments, formats, lengths, publishing times) and to your discovered competitors. "
            "Subscribers' private behaviour (watch time, retention, demographics) is not public, so it is inferred from those proxies. "
            "Your video is analysed temporarily and deleted immediately; only a few sampled frames are sent to the AI provider, and nothing is stored.")
    with st.form("studio_form"):
        up_file = st.file_uploader("Your video", type=["mp4", "mov", "m4v", "webm", "mkv"])
        c1, c2 = st.columns(2)
        title = c1.text_input("Working title (optional)")
        topic = c2.text_input("Topic / main keyword (optional when AI is on)", placeholder="e.g. cold brew coffee at home")
        summary = st.text_area("Notes about the video (optional)", height=70)
        description = st.text_area("Draft description / captions text (optional, for the copyright text screen)", height=60)
        c3, c4, c5 = st.columns(3)
        fmt = c3.radio("Format", ["Auto", "Short", "Long-form"], horizontal=True)
        tz = c4.selectbox("Your time zone", upl.COMMON_TZ, index=1)
        weekday = c5.selectbox("Planned weekday (views estimate)", ["Not decided"] + upl.WEEKDAYS)
        st.markdown("**Rights declaration** (used only for the risk screen)")
        d1, d2 = st.columns(2)
        music = d1.selectbox("Music", ["unsure", "none", "own", "licensed", "unlicensed"], format_func=lambda x: {"unsure": "Not sure where it's from", "none": "No music", "own": "My own music", "licensed": "Licensed / royalty-free with proof", "unlicensed": "Popular/unlicensed track"}[x])
        footage = d2.selectbox("Footage and images", ["unsure", "all_original", "licensed", "others_clips"], format_func=lambda x: {"unsure": "Not sure", "all_original": "All original", "licensed": "Licensed with proof", "others_clips": "Includes others' clips (TV, films, sports, other creators)"}[x])
        e1, e2, e3, e4, e5 = st.columns(5)
        commentary = e1.checkbox("Adds my commentary")
        lyrics = e2.checkbox("Uses lyrics/scripts")
        ai_gen = e3.checkbox("Mostly AI-generated")
        templ = e4.checkbox("Same template as my other videos")
        reup = e5.checkbox("Re-upload of earlier video")
        goal = st.selectbox("Main goal for the call to action", ["subscribers", "watch time", "engagement", "traffic"])
        credit = st.text_input("Music credit for the description (optional)", placeholder="Track name - Artist - licence/link")
        go_btn = st.form_submit_button("🎬 Analyse my video")
    if go_btn:
        if not (title.strip() or topic.strip() or up_file):
            st.error("Upload a video, or add at least a title or topic.")
        else:
            probe, frames, transcript = None, [], None
            try:
                if up_file is not None:
                    with st.spinner("Measuring your video and sampling frames..."):
                        probe, frames, transcript = vprobe.analyze_bytes_full(up_file.getvalue(), "." + up_file.name.rsplit(".", 1)[-1], with_transcript=True)
                with st.spinner("Understanding the video and building your recommendations..."):
                    ref_titles = pd.concat([d["title"] for d in r["competitor_videos"].values()]).tolist() if r["competitor_videos"] else []
                    st.session_state["studio"] = stu.run_studio(
                        r, {"title": title, "topic": topic, "summary": summary, "description": description, "format": fmt, "tz": tz,
                            "weekday": None if weekday == "Not decided" else upl.WEEKDAYS.index(weekday), "music": music, "footage": footage,
                            "commentary": commentary, "lyrics_or_scripts": lyrics, "ai_generated": ai_gen, "templated": templ, "reupload": reup,
                            "goal": goal, "music_credit": credit}, probe, prov, st.session_state.get("trend_terms"), ref_titles, frames=frames, transcript=transcript)
            except vprobe.FFmpegMissing as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))
    S = st.session_state.get("studio")
    if not S:
        st.caption("Upload a video and press **Analyse my video**. Run the Trend Radar first if you want trending words in the titles and hashtags.")
        return
    und = S["understanding"]
    st.success(f"Planned format: **{S['format']}**" + (f" | {S['ai_note']}" if S.get("ai_note") else ""))
    if S["audience_summary"]:
        st.markdown("**What your public data says your audience responds to:** " + " ".join(S["audience_summary"]))

    if S["probe"]:
        st.markdown(section("1. Your video, measured"), unsafe_allow_html=True)
        _fmt_probe_cards(S["probe"])
        tp = S["probe"].get("tempo") or {}
        if tp.get("bpm"):
            st.caption(f"Rough tempo of the audio: {tp['bpm']:.0f} BPM (confidence {tp['confidence']:.2f}; only meaningful when music dominates).")

    st.markdown(section("2. What the AI understood about your video"), unsafe_allow_html=True)
    if und["confidence"] == "none":
        st.info(und["source"] + ". Add an AI key (AI_API_KEY in secrets) to switch on automatic understanding.")
    else:
        st.markdown(card(f"{und['topic'] or 'Topic not detected'}", f"{und['summary']}<br><b>Niche:</b> {und['niche']} | <b>Audience:</b> {und['audience']} | <b>Mood:</b> {und['mood']} | <b>Language:</b> {und['language']}<br>"
                         f"<b>Key moments:</b> {'; '.join(und['key_moments'])}<br><i>{und['source']} - confidence {und['confidence']}. AI can be wrong; check it.</i>", PALETTE["violet"], und["confidence"].title()), unsafe_allow_html=True)

    st.markdown(section("3. Recommended title, description and tags"), unsafe_allow_html=True)
    st.caption("'Channel fit' scores how closely a title matches what has worked on YOUR channel (pattern, length, keyword placement, specificity) and flags risky wording or duplicates. It is not a prediction of views.")
    if len(S["titles"]):
        st.dataframe(S["titles"][["title", "fit_score", "pattern", "source", "notes"]], hide_index=True, width="stretch",
                     column_config={"fit_score": st.column_config.ProgressColumn("Channel fit", min_value=0, max_value=100, format="%.0f")})
    st.markdown("**Hashtags** (first 3 show above the title): " + " ".join(S["hashtags"]["hashtags"]))
    st.caption(S["hashtags"]["note"])
    st.text_area("Tags (hidden, optional)", S["hashtags"]["hidden_tags"], height=60, key="hid_tags")
    st.text_area("Title + description draft (edit freely)", S["caption"], height=230, key="cap_txt")

    st.markdown(section("4. Probable performance: views, subscribers, engagement and watch time"), unsafe_allow_html=True)
    fc, vw = S["forecast"], S["views"]
    if fc.get("available"):
        dur = float((S["probe"] or {}).get("duration_seconds") or (30 if S["format"] == "Short" else 600))
        pct = st.slider("Assumed average % of the video watched (not measured - edit it)", 10, 100, int(round(fc["pct_watched"] * 100)), step=5, key="pct_watched") / 100
        fc = gpred.forecast(r, vw, S["format"], dur, pct)
        t = fc["table"].set_index("metric")
        def mid(m):
            return format_number(t.loc[m, "mid"]) if m in t.index else "n/a"
        cards([("Views (typical)", mid("Views at maturity"), f"range {format_number(t.loc['Views at maturity', 'low'])}-{format_number(t.loc['Views at maturity', 'high'])}"),
               ("New subscribers", mid("New subscribers"), "rough; per-video data is private"),
               ("Likes + comments", format_number((t.loc['Likes', 'mid'] if 'Likes' in t.index else 0) + (t.loc['Comments', 'mid'] if 'Comments' in t.index else 0)), "your typical rates"),
               ("Watch hours", mid("Watch hours (assumption-based)"), f"if {pct:.0%} is watched")])
        st.dataframe(fc["table"].round(1), hide_index=True, width="stretch")
        for line in fc["context"]:
            st.markdown(f"- {line}")
        bt = vw.get("backtest", {})
        if bt.get("n"):
            st.markdown(f"**Back-test on your own videos in this format ({bt['n']}):** typical error {bt['median_abs_pct_error_baseline']:.0%} for the plain median vs {bt['median_abs_pct_error_with_factors']:.0%} with topic/title/weekday/length adjustments; "
                        f"{bt['share_inside_range']:.0%} fell inside the range. Adjustments {'are' if vw['factors_used'] else 'are NOT'} used because they {'beat' if vw['factors_used'] else 'did not beat'} the plain median.")
        if len(vw.get("factors", [])):
            st.dataframe(vw["factors"], hide_index=True, width="stretch")
        for n_ in fc["notes"] + vw["notes"]:
            st.caption(n_)
    else:
        st.info(fc.get("reason", "Not available."))

    st.markdown(section("5. Your upload vs your competitors"), unsafe_allow_html=True)
    if len(S["comparison"]):
        st.dataframe(S["comparison"].round(3), hide_index=True, width="stretch")
        st.caption("Peers = the competitors discovered automatically, same format. 'Typical range' is their 25th-75th percentile; being outside it is a prompt to think, not an error.")
    else:
        st.info("No competitor data to compare with.")
    if len(S["closest"]):
        st.markdown("**Closest competitor videos (most similar topic)**")
        show_df(S["closest"])

    st.markdown(section("6. Recommended upload schedule"), unsafe_allow_html=True)
    pl = S["planner"]
    st.caption(f"Time zone: {pl['tz']}. Reliability: **{pl['reliability']}** - {pl['reliability_note']} Slots are your own mature videos' median views relative to your median, shrunk toward a prior; slots with your own data are preferred.")
    if len(pl["best"]):
        st.markdown("**Best slots to test**")
        st.dataframe(pl["best"][["day", "window", "own_videos", "own_median_rel", "score", "evidence"]], hide_index=True, width="stretch")
    if len(pl["avoid"]):
        st.markdown("**Slots that have underperformed for you (3+ videos)**")
        st.dataframe(pl["avoid"][["day", "window", "own_videos", "score"]], hide_index=True, width="stretch")
    heat = pl["slots"].pivot(index="day", columns="window", values="score").reindex(upl.WEEKDAYS)
    st.plotly_chart(style(px.imshow(heat, color_continuous_scale="Viridis", aspect="auto", labels={"color": "slot score"}), "Slot score by weekday and 3-hour window (your time zone)"), width="stretch")
    if len(pl["plan"]):
        st.markdown(f"**Suggested upload schedule for the next 14 days** (goal ~{pl['uploads_per_30_days_goal']:.1f} uploads per 30 days; first line is this video)")
        show_df(pl["plan"])
        csv_button("Download upload plan (CSV)", pl["plan"], "upload_plan.csv", "dl_upload_plan")

    st.markdown(section("7. Editing and music advice"), unsafe_allow_html=True)
    colors = {"High": PALETTE["coral"], "Medium": PALETTE["violet"], "Low": PALETTE["teal"]}
    for a in S["editing_advice"].itertuples():
        st.markdown(card(f"{a.area}: {a.finding}", f"{a.advice}<br><i>Evidence: {a.evidence}</i>", colors.get(a.priority, PALETTE["violet"]), a.priority), unsafe_allow_html=True)
    m = S["music"]
    extra = f" The AI suggests, from your footage: **{m['ai_mood']}**." if m.get("ai_mood") else ""
    st.markdown(f"**Music direction ({m['niche']} niche):** {m['mood']}, about {m['bpm_low']}-{m['bpm_high']} BPM; genres to explore: {', '.join(m['genres'])}.{extra} {m['licence']} *({m['basis']})*")

    st.markdown(section("8. Copyright and reused-content risk screen"), unsafe_allow_html=True)
    cp = S["copyright"]
    cards([("Risk screen score", f"{cp['risk_score']}/100", "from your declarations, text and AI notes"), ("Level", cp["level"], cp["summary"])])
    if len(cp["findings"]):
        st.dataframe(cp["findings"][["severity", "issue", "why_it_matters", "what_to_do"]], hide_index=True, width="stretch")
    for item in cp["checklist"]:
        st.markdown(f"- [ ] {item}")
    st.warning(cp["disclaimer"])
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.download_button("Download Video Studio report (Excel)", build_studio_workbook(S), "video_studio_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_studio_xlsx")
    c2.download_button("Download Video Studio data (JSON)", results_to_json_bytes(S), "video_studio.json", "application/json", key="dl_studio_json")


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
               ("3  Growth roadmap", "Day by day", "targets from peer KPIs"), ("4  Income", "Estimate + roadmap", "time to monetization"), ("5  Export", "PDF / Excel", "Word, CSV, JSON")])
        st.info("Enter your channel in the sidebar and press **Analyse my channel**. Competitors are found for you automatically.")
        st.markdown("**What public data can't tell you:** watch time, impressions, click-through rate, retention, traffic sources and revenue are private to channel owners. The Income tab shows clearly-labelled *estimates* only, never actual revenue.")
        return
    names = ["Overview", "🚀 Growth Roadmap", "💰 Income & Monetization", "🔥 Trend Radar", "🎬 Video Studio", "Video Performance", "Title Analysis", "Topics", "Competitors", "Content Gaps", "AI Content Plan", "Sources and Export"]
    fns = dict(zip(names, (tab_overview, tab_roadmap, tab_income, tab_trends, tab_studio, tab_videos, tab_titles, tab_topics, tab_competitors, tab_gaps, tab_plan, tab_export)))
    with st.container(key="nav"):  # persistent navigation: unlike st.tabs it keeps your place when a button reruns the app
        active = st.radio("Section", names, horizontal=True, key="active_tab", label_visibility="collapsed")
    fns[active](r)


main()
