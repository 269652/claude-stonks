"""Broker abstraction. The engine only ever talks to this interface, so paper
and live Trade Republic execution are fully interchangeable."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderResult:
    ok: bool
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0
    message: str = ""
    order_id: str | None = None   # exchange order id (live limit orders)


class Broker:
    mode: str = "base"

    def cash(self) -> float:
        raise NotImplementedError

    def price(self, symbol: str) -> float:
        raise NotImplementedError

    def buy(self, symbol: str, qty: float, price: float) -> OrderResult:
        raise NotImplementedError

    def sell(self, symbol: str, qty: float, price: float) -> OrderResult:
        raise NotImplementedError
