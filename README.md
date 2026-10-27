# AI YouTube Channel Analyzer

Analyse a YouTube channel's **public** performance, compare up to five competitors, find content gaps, and plan what to publish next. Built with Streamlit, the official YouTube Data API v3, Pandas and Plotly, with optional AI features.

## Features
- **Automatic competitor discovery**: no competitor input. The app searches recent videos on your own topics, collects the channels behind them, drops yourself and auto-generated "- Topic" channels, and keeps the 10+ best topical and audience-size matches (heuristic, shown with a match score and reason).
- **Growth roadmap**: channel health score (0-100), KPIs vs the discovered peers (uploads per 30 days, upload steadiness, views per day, views per 1,000 subscribers, engagement, breakout rate), planning targets for day 30/60/90, priority actions, a tick-off **day-by-day 30-day plan**, 90-day milestones and a what-if calculator. Targets are planning goals, not predictions.
- **Channel overview**: lifetime public statistics, kept separate from statistics of the selected video sample/date range.
- **Video performance**, **title and thumbnail analysis**, **topics** (editable), **content gaps** (transparent priority score), **AI content plan** (7/30-day calendar).
- **Exports**: PDF, Excel (filters, number formats, links), CSV, JSON, Word - all include the roadmap.
- **Demo mode**: clearly labelled sample data when no API key is configured on the server.
- Colourful, responsive dashboard theme.

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
- Competitor discovery is a topical heuristic; review the list. Very small or niche channels may yield fewer than 10 matches (the app says so).
- Shorts are inferred from duration (3 minutes or less); the API has no Shorts flag.
- Topics use metadata only; transcripts and comment mining are not implemented.
- Content-gap scores and roadmap targets use peer performance, not search volume, and are not forecasts.
- Thumbnail analysis is simple image statistics; text density is not measured.
- AI outputs are recommendations, not guaranteed improvements.
