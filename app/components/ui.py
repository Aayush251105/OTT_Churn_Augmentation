"""Reusable layout pieces: page header, section headings, KPI cards, callouts, tags, empty states.

Every helper renders plain HTML styled by app/assets/style.css, so all pages share one look.
"""

from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def inject_css() -> None:
    """Load the stylesheet once per run (called by the router before the page runs)."""
    st.html(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>")


def page_header(title: str, subtitle: str) -> None:
    st.html(f'<div class="ss-header"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>')


def section(title: str, helper: str | None = None, step: int | None = None) -> None:
    """Small section heading with optional helper text and step number."""
    num = f'<span class="ss-step">{step}</span>' if step else ""
    help_html = f"<p>{escape(helper)}</p>" if helper else ""
    st.html(f'<div class="ss-section"><h2>{num}{escape(title)}</h2>{help_html}</div>')


def chart_title(title: str, caption: str | None = None) -> None:
    """Takeaway title (and caption) above a chart inside a card."""
    cap = f'<p class="ss-chart-caption">{escape(caption)}</p>' if caption else ""
    st.html(f'<p class="ss-chart-title">{escape(title)}</p>{cap}')


def kpi(label: str, value: str, sub: str | None = None, delta: str | None = None, tone: str = "flat",
        small: bool = False) -> str:
    """HTML for one KPI card. tone: up (red, worse), down (teal, better), flat (grey)."""
    d = f'<div class="sub {tone}">{escape(delta)}</div>' if delta else ""
    s = f'<div class="sub">{escape(sub)}</div>' if sub else ""
    cls = "val sm" if small else "val"
    return f'<div class="ss-kpi"><div class="lbl">{escape(label)}</div><div class="{cls}">{escape(value)}</div>{d}{s}</div>'


def kpi_row(cards: list[str]) -> None:
    """Render KPI cards in equal columns."""
    for col, card in zip(st.columns(len(cards)), cards):
        col.html(card)


def callout(body: str, title: str | None = None, tone: str = "grey") -> None:
    """Callout with a coloured left border. tone: teal, red, amber, grey."""
    t = f'<div class="t">{escape(title)}</div>' if title else ""
    st.html(f'<div class="ss-callout {tone}">{t}<div class="b">{escape(body)}</div></div>')


def tags(items) -> None:
    st.html('<div class="ss-tags">' + "".join(f'<span class="ss-tag">{escape(str(i))}</span>' for i in items) + "</div>")


def badge(tier: str) -> str:
    return f'<span class="ss-badge {escape(tier)}">{escape(tier.upper())}</span>'


def numbered_list(rows: list[tuple[str, str, str]]) -> None:
    """Rows of (heading, description, right-aligned note) with automatic numbering."""
    html = "".join(
        f'<div class="row"><div class="n">{i:02d}</div><div><div class="h">{escape(h)}</div>'
        f'<div class="d">{escape(d)}</div></div><div class="r">{escape(r)}</div></div>'
        for i, (h, d, r) in enumerate(rows, 1))
    st.html(f'<div class="ss-list">{html}</div>')


def empty_state(title: str = "Results not generated yet", hint: str = "") -> None:
    st.html(f'<div class="ss-empty"><div class="t">{escape(title)}</div><div>{escape(hint)}</div></div>')


def banner(text: str) -> None:
    st.html(f'<div class="ss-banner">{escape(text)}</div>')


# ---------------------------------------------------------------- number formatting
def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def num(x: float) -> str:
    return f"{x:,.0f}"


def money(x: float) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def signed(x: float, digits: int = 3) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(x):.{digits}f}"
