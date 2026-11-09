"""V9.5 presentation-only helpers. The V8.4 engine remains authoritative."""
from __future__ import annotations
import math


def sort_sectors_by_exposure(sectors: dict) -> list[str]:
    """Largest verified sector first. Unknown/uncategorized stays last."""
    def key(k: str):
        if str(k).strip().lower() in ('unknown','unclassified','not_applicable'):
            return (1, 0.0, str(k))
        rec = sectors.get(k) if isinstance(sectors.get(k), dict) else {}
        val = rec.get('portfolio_fraction')
        num = float(val) if isinstance(val, (int,float)) and math.isfinite(val) else 0.0
        return (0, -num, str(k))
    return sorted(sectors, key=key)


def coverage_summary(known_pct: float, unknown_pct: float) -> tuple[str,str]:
    """No renormalization. Gold can be classified as a commodity, not a sector."""
    if not (math.isfinite(known_pct) and math.isfinite(unknown_pct)):
        return 'Sector coverage unavailable', 'Not enough data to explain the unclassified portion.'
    return (f'{known_pct:.2f}% in named categories',
            f'{unknown_pct:.2f}% has no relevant category or is not classified. '
            'For example, a gold investment is a known commodity but does not belong to a company sector.')
