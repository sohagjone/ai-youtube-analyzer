# AI YouTube Channel Analyzer

Analyse a YouTube channel's **public** performance, compare up to five competitors, find content gaps, and plan what to publish next. Built with Streamlit, the official YouTube Data API v3, Pandas and Plotly, with optional AI features.

## Features
- **Automatic competitor discovery**: no competitor input. The app searches recent videos on your own topics, collects the channels behind them, drops yourself and auto-generated "- Topic" channels, and keeps the 10+ best topical and audience-size matches (heuristic, shown with a match score and reason).
- **Growth roadmap**: channel health score (0-100), KPIs vs the discovered peers (uploads per 30 days, upload steadiness, views per day, views per 1,000 subscribers, engagement, breakout rate), planning targets for day 30/60/90, priority actions, a tick-off **day-by-day 30-day plan**, 90-day milestones and a what-if calculator. Targets are planning goals, not predictions.
- **Income & Monetization tab (estimates)**: paste a channel link and see (1) estimated income now, (2) how long until it can join the YouTube Partner Program (subscribers + public watch hours or Shorts views, using the rule in force each month, including the 2027-02-01 increase to 8,000 hours / 20M Shorts views), and (3) an income roadmap: milestones, 24-month projection for three scenarios, and what you need per month to qualify in 6 or 12 months. Assumptions (niche income per 1,000 views, % of video watched, growth, your real subscriber count, monetization status) are editable.
- **Channel overview**: lifetime public statistics, kept separate from statistics of the selected video sample/date range.
- **Video performance**, **title and thumbnail analysis**, **topics** (editable), **content gaps** (transparent priority score), **AI content plan** (7/30-day calendar).
- **Exports**: PDF, Excel (filters, number formats, links), CSV, JSON, Word - all include the roadmap.
- **Demo mode**: clearly labelled sample data when no API key is configured on the server.
- Colourful, responsive dashboard theme.

## Trend Radar and Video Studio
**🔥 Trend Radar** - pick a niche (Football, Gaming, Tech, Food...) or type your own keywords and the app searches the platforms the owner has connected, **using official APIs only (no scraping)**, then ranks the fastest-moving videos with a per-platform *viral score* (a percentile within that platform's results, because metrics differ), hashtags used by the high scorers, and what the fastest videos share. Also: YouTube's official most-popular chart and a creator watch-list.

| Platform | Needs (in Streamlit secrets) | Notes |
|---|---|---|
| YouTube | `YOUTUBE_API_KEY` | search.list + videos.list (about 200 quota units per search) |
| X | `X_BEARER_TOKEN` | recent search (last 7 days), `has:videos`; needs an API plan that allows search |
| TikTok | `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET` | **Research API - only for approved research projects.** Ordinary developer keys and creator/commercial accounts get HTTP 401 (`access_token_invalid` / missing `research.data.basic`). Without approval use TikTok's free Creative Center and paste its hashtags into the Trend Radar box |
| Instagram | `META_ACCESS_TOKEN`, `INSTAGRAM_BUSINESS_ACCOUNT_ID` | hashtag search (popular posts, no view counts; 30 unique hashtags per 7 days; needs Instagram Public Content Access) |
| Facebook | - | no public video search API exists (CrowdTangle was shut down), so it is not scraped |

The X, TikTok and Instagram connectors follow those platforms' published API formats and are covered by mocked-request tests, but **could not be run against the live APIs here** - try each with your credentials and tell me if a platform rejects a request. Unconnected platforms simply show "not connected".

**🎬 Video Studio** - upload a video; the app analyses it automatically.
1. **Measured facts** (bundled ffmpeg via `imageio-ffmpeg`): loudness (LUFS), peak, silence, cuts per minute, hook, framing, rough tempo. The file is deleted immediately.
2. **AI understanding** (needs `AI_API_KEY`): a few sampled frames (plus an optional local transcript if `faster-whisper` is installed) go to your AI provider, which returns the topic, summary, key moments, mood, draft titles/description/tags and music mood. It is told not to identify people or invent facts, and to flag possible rights issues as uncertain. **Without an AI key the app cannot see what your video is about** and uses only the text you type.
3. **Recommended title, description, tags and hashtags**, titles ranked by *channel fit* (your best pattern, length, keyword placement, risky wording, duplicates).
4. **Probable performance**: views (from your own history in that format, with a back-test), **new subscribers** (views x your lifetime subscribers-per-view ratio, range from similar channels), likes/comments (your own rates) and **watch hours** (rests on an *editable average-%-watched assumption* because watch time is not public). These are ranges and rough estimates, not forecasts.
5. **Your upload vs competitors**: length, title, expected engagement and size-adjusted views against the discovered peers' typical range, plus the closest competitor videos.
6. **Upload schedule** from your own mature videos' best weekday/time slots (your time zone) and a 14-day plan.
7. **Editing and music advice** (measured, from your channel data, or rule of thumb - labelled) and a copyright/reused-content **risk screen**.

Honest limits: "subscribers' behaviour" is inferred from **public** proxies (retention and demographics are private). The copyright screen **cannot** detect Content ID matches and is not legal advice - upload as Private and read YouTube Studio's *Checks*.

Navigation uses persistent pills (not `st.tabs`), so pressing a button keeps you on the section you are working in.

## What public data cannot show
Watch time, impressions, click-through rate, retention, traffic sources and revenue are private to channel owners. The app never shows or estimates them. Optional YouTube Analytics OAuth is **not implemented**.

## API key (end users are never asked for one)
The key is read from `YOUTUBE_API_KEY` in `.streamlit/secrets.toml` (local, git-ignored) or in Streamlit Cloud secrets, or from an environment variable. It is never hard-coded in source. Set `USE_SAMPLE_DATA=1` to force demo mode.

| Name | Required | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | For live data | YouTube Data API v3 key |
| `AI_PROVIDER`, `AI_API_KEY`, `AI_MODEL` | Optional | Enables AI title/plan refinement (`anthropic` or `openai`) |

**Security:** never upload `.streamlit/secrets.toml` to GitHub. In Google Cloud, restrict the key to YouTube Data API v3 and rotate it if it was ever shared publicly.

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# .streamlit/secrets.toml already holds the key (git-ignored); otherwise copy secrets.toml.example
streamlit run app.py
```
With no key configured the app starts in demo mode.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
Tests use mocked HTTP and illustrative fixtures; no network or real keys.

## Publish to GitHub
```bash
git init
git add .
git commit -m "Initial commit: AI YouTube Channel Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-youtube-channel-analyzer.git
git push -u origin main
```
Confirm `.streamlit/secrets.toml` is NOT staged (`git status`). If you upload files through the GitHub web page, do not include the `.streamlit/secrets.toml` file.

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Create app, choose the repository and branch `main`, main file `app.py`.
3. Advanced settings: pick Python 3.11 or 3.12; paste your secrets (the same `YOUTUBE_API_KEY = "..."` line) so users never need a key.
4. Deploy, then check the logs for dependency errors.

## API quota
Default quota is 10,000 units/day. One analysis costs roughly 400 units for competitor discovery (4 searches x 100) plus about 3 units per 50 videos per channel (10 competitors at 40 videos is about 60 units), so roughly 450-500 units, or about 20 analyses per day. Custom `/c/` URLs may add a 100-unit search. Quota resets daily (Pacific Time).

## Known limitations
- Video file analysis uses at most the first 3 minutes for cuts and tempo; very large files need memory (upload limit is 200 MB in `.streamlit/config.toml`).
- Tempo estimation is only meaningful when music dominates the audio.
- Competitor discovery is a topical heuristic; review the list. Very small or niche channels may yield fewer than 10 matches (the app says so).
- Shorts are inferred from duration (3 minutes or less); the API has no Shorts flag.
- Topics use metadata only; transcripts and comment mining are not implemented.
- Content-gap scores and roadmap targets use peer performance, not search volume, and are not forecasts.
- Thumbnail analysis is simple image statistics; text density is not measured.
- AI outputs are recommendations, not guaranteed improvements.
