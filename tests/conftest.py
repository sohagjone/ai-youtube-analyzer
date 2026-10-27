import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def now():
    return NOW


@pytest.fixture(scope="session")
def results(now):
    """Full analysis on ILLUSTRATIVE sample data (not real YouTube data)."""
    from src.content_strategy import generate_calendar
    from src.pipeline import run_analysis
    from src.sample_data import SampleClient

    r = run_analysis(SampleClient(now), "@sample_own", ["@rival_a", "@rival_b"], sample_size=40, now=now)
    r["calendar"] = generate_calendar(r["gaps"], r["topics"], days=7, posts_per_week=3, start=now.date())
    return r
