"""lab_data: the Northwind Gear shop data every notebook shares, derived deterministically from the dbt seeds."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"))
import lab_data  # noqa: E402

COUNTRY_BOX = lab_data.COUNTRY_BOXES


def test_seeds_load():
    assert len(lab_data.customers()) == 20
    assert len(lab_data.products()) == 10
    assert len(lab_data.orders()) == 60
    assert {"payment_id", "order_id", "method", "amount"} <= set(lab_data.payments()[0])


def test_everything_is_deterministic():
    for fn in (lab_data.clickstream, lab_data.deliveries, lab_data.order_events, lab_data.product_docs):
        assert fn() == fn(), fn.__name__


def test_clickstream_references_real_entities():
    customers = {c["customer_id"] for c in lab_data.customers()}
    products = {p["product_id"] for p in lab_data.products()}
    for e in lab_data.clickstream():
        assert e["customer_id"] in customers
        if "product_id" in e["payload"]:
            assert e["payload"]["product_id"] in products


def test_one_checkout_per_order_at_order_time():
    orders = {o["order_id"]: o for o in lab_data.orders()}
    checkouts = [e for e in lab_data.clickstream() if e["kind"] == "checkout"]
    assert Counter(e["payload"]["order_id"] for e in checkouts) == Counter({oid: 1 for oid in orders})
    for e in checkouts:
        o = orders[e["payload"]["order_id"]]
        assert e["customer_id"] == o["customer_id"] and e["payload"]["product_id"] == o["product_id"]
        assert e["event_ts"] <= o["order_ts"]


def test_payload_shape_differs_by_kind():
    keys = {}
    for e in lab_data.clickstream():
        keys.setdefault(e["kind"], set()).update(e["payload"])
    assert set(keys) == {"page_view", "search", "add_to_cart", "checkout"}
    assert "query" in keys["search"] and "query" not in keys["checkout"]
    assert "order_id" in keys["checkout"] and "order_id" not in keys["page_view"]


def test_deliveries_one_per_order_inside_the_country():
    country = {c["customer_id"]: c["country"] for c in lab_data.customers()}
    orders = {o["order_id"]: o for o in lab_data.orders()}
    rows = lab_data.deliveries()
    assert sorted(r["order_id"] for r in rows) == sorted(orders)
    for r in rows:
        lat0, lat1, lon0, lon1 = COUNTRY_BOX[country[orders[r["order_id"]]["customer_id"]]]
        assert lat0 <= r["lat"] <= lat1 and lon0 <= r["lon"] <= lon1


def test_order_events_replay_every_order_in_time_order():
    events = lab_data.order_events()
    assert sorted(e["order_id"] for e in events) == sorted(o["order_id"] for o in lab_data.orders())
    assert [e["order_ts"] for e in events] == sorted(e["order_ts"] for e in events)
    assert {e["status"] for e in events} <= {"placed", "shipped", "returned"}   # raw status values


def test_product_docs_cover_every_product():
    docs = lab_data.product_docs()
    for p in lab_data.products():
        kinds = {d["kind"] for d in docs if d["product_id"] == p["product_id"]}
        assert {"manual", "review"} <= kinds, p["name"]


def test_support_tickets_follow_the_refunds():
    """Tickets mirror the refunds seed: routers are returned most, SSDs arrive damaged most."""
    name = {p["product_id"]: p["name"] for p in lab_data.products()}
    tickets = [d for d in lab_data.product_docs() if d["kind"] == "ticket"]
    per_product = Counter(name[t["product_id"]] for t in tickets)
    assert per_product.most_common(2)[0][0] in {"router", "ssd"}
    assert {"router", "ssd"} <= {n for n, _ in per_product.most_common(3)}
    damaged = Counter(name[t["product_id"]] for t in tickets if t["topic"] == "damaged")
    assert damaged.most_common(1)[0][0] == "ssd"
    assert all(t["order_id"] for t in tickets)        # every ticket is about a real refunded order
