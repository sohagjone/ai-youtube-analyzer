# AI YouTube Channel Analyzer

Analyse a YouTube channel's **public** performance, compare up to five competitors, find content gaps, and plan what to publish next. Built with Streamlit, the official YouTube Data API v3, Pandas and Plotly, with optional AI features.

## Features
- **Channel overview**: lifetime public statistics, kept separate from statistics of the selected video sample/date range.
- **Video performance**: mean/median views, views per day, performance relative to your median, engagement indicators, format (short vs long) comparison. Calculations are documented in-app.
- **Title and thumbnail analysis**: length, numbers, questions, keyword placement, repeated structures, near-duplicates, wording to review, three title alternatives per video (rule-based, or AI if configured), optional thumbnail image statistics.
- **Topics**: keyword-based grouping from titles/tags/descriptions, with custom keyword mapping and manual corrections.
- **Competitors**: side-by-side table and charts with size-adjusted caveats.
- **Content gaps**: ranked opportunities with a transparent score (30% relevance, 30% demand evidence, 20% differentiation, 20% feasibility), confidence level, evidence, suggested title, format and outline.
- **AI content plan**: 7- or 30-day calendar; each row is tied to evidence or labelled a creative hypothesis.
- **Keyword search** via the official API.
- **Exports**: PDF, Excel (filters, number formats, clickable links), CSV, JSON, Word.
- **Sample-data mode**: clearly labelled illustrative data when no key is available or quota is exhausted.

## What public data cannot show
Watch time, impressions, click-through rate, retention, traffic sources and revenue are private to channel owners. The app never shows or estimates them. Optional YouTube Analytics OAuth is **not implemented** in this MVP.

## Required credentials
| Name | Required | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | For live data | YouTube Data API v3 key |
| `AI_PROVIDER`, `AI_API_KEY`, `AI_MODEL` | Optional | `anthropic` (default) or `openai`; leave model empty for the default |

Get a YouTube key: Google Cloud Console, create a project, enable **YouTube Data API v3**, create an API key (restrict it to that API).

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it (git-ignored)
streamlit run app.py
```
With no key, tick the SAMPLE-data option in the sidebar.

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
Confirm `.streamlit/secrets.toml` is not staged (`git status`).

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Create app, choose the repository and branch `main`, main file `app.py`.
3. Advanced settings: pick Python 3.11 or 3.12; paste your secrets (same format as `secrets.toml.example`).
4. Deploy, then check the logs for dependency errors.

## API quota
Default quota is 10,000 units/day. Per channel: about 1 + 2 per 50 videos (e.g. 3 units for 50 videos). Keyword search costs 100 units per call. Custom `/c/` URLs and bare names may cost an extra 100-unit search. Quota resets daily (Pacific Time).

## Known limitations
- Shorts are inferred from duration (3 minutes or less); the API has no Shorts flag.
- Topics use metadata only; transcripts and comment mining are not implemented.
- Content-gap scores use competitor performance, not search volume.
- Search results are not a complete or demand-ranked list.
- Thumbnail analysis is simple image statistics; text density is not measured.
- AI outputs are recommendations, not guaranteed improvements.
