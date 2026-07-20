"""Ledger entries for settling orders with the payment provider."""

from dataclasses import dataclass

DISCOUNT_RATE = 0.1


@dataclass
class LedgerEntry:
    reference: str
    amount_cents: int
    settled: bool = False


class Ledger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def record(self, reference: str, amount_cents: int) -> LedgerEntry:
        entry = LedgerEntry(reference=reference, amount_cents=amount_cents)
        self._entries.append(entry)
        return entry

    def apply_discount(self, entry: LedgerEntry) -> int:
        return int(entry.amount_cents * (1 - DISCOUNT_RATE))

    def settle(self, entry: LedgerEntry) -> None:
        entry.settled = True
