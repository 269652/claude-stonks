"""Cash buckets: a reserve for the user's manual actions, a dip budget for
averaging down held positions, and the rest for new entries."""
from __future__ import annotations


def split_cash(cash: float, reserve_eur: float, dips_eur: float,
               dip_spent: float = 0.0) -> dict:
    """{"entry", "dips", "reserve"} available now. The reserve is filled
    first, then what is left of the dip budget (minus dip buys still held),
    and entries get the rest."""
    cash = max(0.0, cash)
    reserve = min(cash, max(0.0, reserve_eur))
    dips = min(max(0.0, dips_eur - max(0.0, dip_spent)), cash - reserve)
    return {"entry": max(0.0, cash - reserve - dips), "dips": dips, "reserve": reserve}
