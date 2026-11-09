"""Historical window selection for the GUI; not a return or price calculation."""
from __future__ import annotations

from datetime import date, timedelta

# Both SPY and RSP are required by V8.4 benchmark reports. RSP's history starts
# after 2000, so fetching from this date allows their full joint overlap. It
# does NOT imply data exists from 2000 for any given holding.
AUTO_FETCH_START = date(2000, 1, 1)


def requested_window(auto: bool, manual_start: date, manual_end: date, today: date) -> tuple[date, date]:
    """Request all benchmark-compatible history, or honor exact manual dates."""
    if auto:
        return AUTO_FETCH_START, today + timedelta(days=1)
    return manual_start, manual_end


def actual_window_label(report: dict) -> str:
    """Use observed shared dates, not an arbitrary requested provider window."""
    start, end = report.get("first_date"), report.get("last_date")
    if not start or not end:
        return "Shared history dates unavailable"
    return f"{start} to {end}"
