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


def test_dashboard_filters_archived(live_server, tmp_path: Path, page: Page):
    url, _ = live_server
    _seed(url, tmp_path, name="keep")
    _seed(url, tmp_path, name="hidden")
    # Archive one run via the API (server fixture exports ENDLEX_TOKEN=e2e-tok).
    import httpx
    with httpx.Client(base_url=url) as c:
        r = c.patch(
            "/api/runs/hidden/state",
            json={"archived": True, "tags": ["wip"]},
            headers={"Authorization": "Bearer e2e-tok"},
        )
        assert r.status_code == 200

    page.goto(url)
    # By default, archived rows are hidden.
    visible = page.locator("tbody tr:visible")
    expect(visible).to_have_count(1)
    expect(visible.first.locator("a")).to_have_text("keep")

    # Toggle: now both should show.
    page.locator("#show-archived").check()
    expect(page.locator("tbody tr:visible")).to_have_count(2)
    # And the archived row carries the chip.
    archived_row = page.locator('tbody tr[data-name="hidden"]')
    expect(archived_row.locator(".chip.archived")).to_be_visible()
    # And the tags chip.
    expect(archived_row.locator(".chip", has_text="wip")).to_be_visible()


def test_dashboard_search_filter(live_server, tmp_path: Path, page: Page):
    """Exercise the filter grammar end-to-end."""
    url, _ = live_server
    # Seed three runs with distinct metric ranges.
    import httpx

    from endlex import Tracker

    def seed_with_loss(name: str, loss: float, tags: list[str]):
        t = Tracker(
            project="proj",
            name=name,
            config={},
            local_dir=tmp_path / f"local-{name}",
            url=url,
            token="e2e-tok",
            batch_size=1,
            batch_interval=0.05,
        )
        try:
            t.log({"step": 0, "train/loss": loss, "val/bpb": loss / 5.0})
        finally:
            t.finish(timeout=5)
        with httpx.Client(base_url=url) as c:
            r = c.patch(
                f"/api/runs/{name}/state",
                json={"tags": tags},
                headers={"Authorization": "Bearer e2e-tok"},
            )
            assert r.status_code == 200

    seed_with_loss("alpha-good", 0.5, ["best"])
    seed_with_loss("alpha-mid", 2.0, ["wip"])
    seed_with_loss("beta-bad", 5.0, ["wip"])

    page.goto(url)
    expect(page.locator("tbody tr:visible")).to_have_count(3)

    # 1. Substring match.
    page.locator("#search").fill("alpha")
    expect(page.locator("tbody tr:visible")).to_have_count(2)

    # 2. Tag filter.
    page.locator("#search").fill("tag:best")
    expect(page.locator("tbody tr:visible")).to_have_count(1)
    expect(page.locator("tbody tr:visible").locator("a").first).to_have_text(
        "alpha-good"
    )

    # 3. Numeric metric filter.
    page.locator("#search").fill("train/loss<1.0")
    expect(page.locator("tbody tr:visible")).to_have_count(1)

    # 4. AND-combined (substring + numeric).
    page.locator("#search").fill("alpha train/loss<1.0")
    expect(page.locator("tbody tr:visible")).to_have_count(1)

    # 5. Clear filter → all back.
    page.locator("#search").fill("")
    expect(page.locator("tbody tr:visible")).to_have_count(3)


def test_compare_view_overlays_runs(live_server, tmp_path: Path, page: Page):
    url, _ = live_server
    _seed(url, tmp_path, name="alpha")
    _seed(url, tmp_path, name="beta")

    page.goto(f"{url}/compare?runs=alpha,beta")
    expect(page).to_have_title(re.compile(r"Compare"))
    # status text updates after first poll fires
    expect(page.locator("#status")).to_contain_text("updated", timeout=15_000)
    # at least one chart materialized with 2 datasets (legend will show both)
    canvases = page.locator("#charts canvas")
    canvases.first.wait_for(state="visible", timeout=10_000)
    assert canvases.count() >= 4
    page.screenshot(path="/tmp/endlex-compare.png", full_page=True)


def test_dashboard_compare_button_navigates(live_server, tmp_path: Path, page: Page):
    url, _ = live_server
    _seed(url, tmp_path, name="alpha")
    _seed(url, tmp_path, name="beta")
    page.goto(url)
    # Compare button starts disabled
    expect(page.locator("#btn-compare")).to_be_disabled()
    # Check both rows
    page.locator('input.row-check[data-name="alpha"]').check()
    page.locator('input.row-check[data-name="beta"]').check()
    expect(page.locator("#btn-compare")).to_be_enabled()
    expect(page.locator("#btn-compare")).to_have_text("Compare selected (2)")
    page.locator("#btn-compare").click()
    expect(page).to_have_url(re.compile(r"/compare\?runs="))


def test_static_export_renders_offline_charts(
    live_server, tmp_path: Path, page: Page
):
    """The static export page renders charts from embedded data — no fetch()."""
    url, _ = live_server
    _seed(url, tmp_path, name="exp")

    # Save the HTML locally and load it via file:// to prove it's self-contained.
    import httpx
    out = tmp_path / "report.html"
    with httpx.Client(base_url=url) as c:
        r = c.get("/api/runs/exp/export.html")
        assert r.status_code == 200
        out.write_text(r.text)

    page.goto(out.as_uri())
    expect(page).to_have_title(re.compile(r"exp.*Endlex"))
    canvases = page.locator("#charts canvas")
    # Wait for Chart.js to load + render.
    canvases.first.wait_for(state="visible", timeout=10_000)
    assert canvases.count() >= 4
    has_pixels = page.evaluate(
        """() => {
          const c = document.querySelector('#charts canvas');
          if (!c) return false;
          const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          for (let i = 3; i < d.length; i += 4) if (d[i] > 0) return true;
          return false;
        }"""
    )
    assert has_pixels


def test_run_page_upgrades_to_sse(live_server, tmp_path: Path, page: Page):
    """After the initial bulk poll, the page upgrades to an EventSource and
    the status text picks up the (live) suffix."""
    url, _ = live_server
    _seed(url, tmp_path, name="live")
    page.goto(f"{url}/run/live")
    # Initial poll lands first ("bytes consumed · updated …").
    expect(page.locator("#status")).to_contain_text("bytes consumed", timeout=10_000)
    # Once SSE attaches and an event arrives, the suffix flips to "(live)".
    # If the stream attaches before any new event, the suffix never appears
    # — push one more metric to guarantee a tick.
    import httpx
    with httpx.Client(base_url=url) as c:
        c.post(
            "/api/runs/live/metrics",
            json=[{"step": 999, "train/loss": 0.1}],
            headers={"Authorization": "Bearer e2e-tok"},
        )
    expect(page.locator("#status")).to_contain_text("(live)", timeout=10_000)


def test_tag_chip_editor(live_server, tmp_path: Path, page: Page):
    """The inline chip editor adds tags via Enter, removes via × and Backspace,
    and persists via the existing PATCH /state endpoint."""
    url, _ = live_server
    _seed(url, tmp_path, name="r")
    page.add_init_script("localStorage.setItem('endlex_token', 'e2e-tok')")
    page.goto(f"{url}/run/r")

    page.locator("#btn-edit-tags").click()
    expect(page.locator("#tags-editor")).to_be_visible()
    # Add two tags via Enter, one via comma.
    page.locator("#tags-input").fill("best")
    page.locator("#tags-input").press("Enter")
    page.locator("#tags-input").fill("exp-1")
    page.locator("#tags-input").press("Enter")
    page.locator("#tags-input").fill("draft")
    page.locator("#tags-input").press(",")  # comma commits
    expect(page.locator("#tags-editor-chips .chip")).to_have_count(3)

    # Backspace on empty input removes the last chip.
    page.locator("#tags-input").press("Backspace")
    expect(page.locator("#tags-editor-chips .chip")).to_have_count(2)

    page.locator("#btn-tags-save").click()
    # Page reloads after success — the persisted chips should reappear.
    expect(page.locator("#tags-display .chip")).to_have_count(2)
    expect(page.locator("#tags-display .chip", has_text="best")).to_be_visible()
    expect(page.locator("#tags-display .chip", has_text="exp-1")).to_be_visible()


def test_run_page_notes_edit_persists(live_server, tmp_path: Path, page: Page):
    """Type notes, save, reload — they should still be there."""
    url, _ = live_server
    _seed(url, tmp_path, name="r")
    page.add_init_script("localStorage.setItem('endlex_token', 'e2e-tok')")
    page.goto(f"{url}/run/r")
    # Open the Notes <details>.
    page.locator("summary", has_text="Notes").click()
    page.locator("#notes").fill("hypothesis: lr too high")
    page.locator("#btn-save-notes").click()
    expect(page.locator("#notes-status")).to_contain_text("saved", timeout=5_000)
    # Reload and confirm persistence.
    page.reload()
    page.locator("summary", has_text="Notes").click()
    expect(page.locator("#notes")).to_have_value("hypothesis: lr too high")


def test_run_page_archive_button_works(live_server, tmp_path: Path, page: Page):
    url, _ = live_server
    _seed(url, tmp_path, name="r")
    # Seed the saved token so the in-page authedFetch doesn't prompt.
    page.add_init_script("localStorage.setItem('endlex_token', 'e2e-tok')")
    page.goto(f"{url}/run/r")
    expect(page.locator("#btn-archive")).to_have_text("Archive")
    page.locator("#btn-archive").click()
    # Page reloads after success; new label should read "Unarchive".
    expect(page.locator("#btn-archive")).to_have_text("Unarchive", timeout=5000)


def test_dashboard_remains_usable_at_narrow_viewport(
    live_server, tmp_path: Path, browser
):
    """At ~mobile widths the table can scroll horizontally and key elements
    (brand, run links) stay visible."""
    url, _ = live_server
    _seed(url, tmp_path, name="alpha")

    ctx = browser.new_context(viewport={"width": 420, "height": 800})
    page = ctx.new_page()
    page.goto(url)
    expect(page.locator("a.brand")).to_be_visible()
    expect(page.get_by_role("link", name="alpha")).to_be_visible()
    # Table is in an overflow-x:auto wrapper so it doesn't break layout.
    wrapper = page.locator(".table-scroll").first
    assert wrapper.evaluate("el => getComputedStyle(el).overflowX") == "auto"
    page.screenshot(path="/tmp/endlex-dashboard-narrow.png", full_page=True)
    ctx.close()


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
