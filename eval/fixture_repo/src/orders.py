"""Order creation and validation."""

from dataclasses import dataclass


@dataclass
class Order:
    order_id: str
    customer_id: str
    total_cents: int


def create_order(order_id: str, customer_id: str, items: list[dict]) -> Order:
    """Create an order from a list of {"price_cents", "quantity"} items."""
    total_cents = sum(item["price_cents"] * item["quantity"] for item in items)
    return Order(order_id=order_id, customer_id=customer_id, total_cents=total_cents)
