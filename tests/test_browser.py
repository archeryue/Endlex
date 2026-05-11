"""Headless Chromium smoke test for the dashboard + run page.

Verifies the dashboard JS that curl-based tests can't:

* Chart.js loads and the run page's poll() loop fires.
* At least one canvas is actually drawn to (non-transparent pixels).
* The dashboard table renders a row per run.

Run with ``uv run pytest tests/test_browser.py``. Chromium must be
installed (``uv run playwright install chromium``). Screenshots land in
``/tmp/endlex-*.png`` for human review.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, expect  # noqa: E402

from endlex import Tracker  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("ENDLEX_SKIP_BROWSER") == "1",
    reason="ENDLEX_SKIP_BROWSER=1 set; skipping browser test",
)


def _seed(url: str, tmp_path: Path, name: str = "demo") -> None:
    t = Tracker(
        project="proj",
        name=name,
        config={"lr": 1e-4, "seed": 42},
        local_dir=tmp_path / "local",
        url=url,
        token="e2e-tok",
        batch_size=5,
        batch_interval=0.1,
    )
    try:
        for step in range(20):
            t.log(
                {
                    "step": step,
                    "train/loss": 9.0 - step * 0.3,
                    "train/mfu": 0.4,
                    "train/dt": 0.3,
                    "train/tok_per_sec": 12345.6,
                    "train/lrm": 1.0,
                }
            )
    finally:
        t.finish(timeout=10)


def test_run_page_renders_and_polls(live_server, tmp_path: Path, page: Page):
    url, _ = live_server
    _seed(url, tmp_path)

    page.goto(f"{url}/run/demo")
    expect(page).to_have_title(re.compile(r"demo.*Endlex"))

    # poll() fires immediately on load and updates #status. Proves the
    # JS loaded, Chart.js loaded, and the fetch round-trip works.
    expect(page.locator("#status")).to_contain_text(
        "bytes consumed", timeout=15_000
    )

    # Multiple panels should have materialized (we logged 5 keys).
    canvases = page.locator("#charts canvas")
    canvases.first.wait_for(state="visible", timeout=10_000)
    count = canvases.count()
    assert count >= 4, f"expected ≥4 chart canvases, got {count}"

    # Chart.js drew non-transparent pixels into the first canvas.
    has_pixels = page.evaluate(
        """() => {
          const c = document.querySelector('#charts canvas');
          if (!c) return false;
          const ctx = c.getContext('2d');
          const d = ctx.getImageData(0, 0, c.width, c.height).data;
          for (let i = 3; i < d.length; i += 4) if (d[i] > 0) return true;
          return false;
        }"""
    )
    assert has_pixels, "chart canvas was empty"

    page.screenshot(path="/tmp/endlex-run-page.png", full_page=True)


def test_dashboard_lists_runs(live_server, tmp_path: Path, page: Page):
    url, _ = live_server
    _seed(url, tmp_path, name="alpha")
    _seed(url, tmp_path, name="beta")

    page.goto(url)
    expect(page).to_have_title("Endlex")
    expect(page.locator("table tbody tr")).to_have_count(2)
    expect(page.get_by_role("link", name="alpha")).to_be_visible()
    expect(page.get_by_role("link", name="beta")).to_be_visible()
    page.screenshot(path="/tmp/endlex-dashboard.png", full_page=True)

    # Sortable headers: clicking "Name" should sort alphabetically ascending,
    # so the first body row's name link should be "alpha".
    page.locator("th.sortable", has_text="Name").click()
    first = page.locator("tbody tr").first.locator("a")
    expect(first).to_have_text("alpha")

    # Clicking the run link navigates to the run page.
    first.click()
    expect(page).to_have_url(re.compile(r"/run/alpha$"))


def test_dashboard_dark_mode(live_server, tmp_path: Path, browser):
    """Sanity-check that the dark-mode palette renders without breaking layout."""
    url, _ = live_server
    _seed(url, tmp_path, name="dark")
    ctx = browser.new_context(color_scheme="dark")
    page = ctx.new_page()
    page.goto(url)
    expect(page).to_have_title("Endlex")
    expect(page.locator("table tbody tr")).to_have_count(1)
    page.screenshot(path="/tmp/endlex-dashboard-dark.png", full_page=True)
    page.goto(f"{url}/run/dark")
    expect(page.locator("#status")).to_contain_text("bytes consumed", timeout=15_000)
    page.screenshot(path="/tmp/endlex-run-page-dark.png", full_page=True)
    ctx.close()
