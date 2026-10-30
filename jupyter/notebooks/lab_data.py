"""Northwind Gear: the one shop every notebook works on.

The dbt seeds (dbt/lakehouse_demo/seeds) are the system of record: customers, products, orders, payments, refunds,
product costs and sales regions. Everything else a notebook needs is derived here from them, deterministically (a
fixed random seed, no wall clock), so every run on every machine sees the same data and every derived row points at
real customers, products and orders:

  clickstream()    web events before and around each order (page views, searches, add-to-cart, checkout);
                   the payload's shape differs by event kind (iceberg/02, VARIANT)
  deliveries()     where each order was delivered, as lat/lon in the customer's country (iceberg/01, clustering)
  order_events()   the orders as they arrived, in time order, with the raw status (iceberg/04, streaming)
  product_docs()   manuals, reviews and support tickets per product; the tickets follow the refunds seed
                   (routers have a connectivity defect and are returned most, SSDs arrive damaged) (rag_demo)

Layers: raw = dbt_demo_raw (seeds), bronze = shop_bronze (Spark, format v2, Trino reads it), shop_v3 = format-v3
feature demos (Spark only), models = dbt_demo. Standard library only, so it imports anywhere.
"""
import csv
import random
from datetime import datetime, timedelta
from functools import lru_cache

from lab_utils import DBT_PROJECT

SEEDS = DBT_PROJECT / "seeds"
_TS = "%Y-%m-%d %H:%M:%S"

# lat_min, lat_max, lon_min, lon_max, and the delivery hubs (city, lat, lon) per customer country
COUNTRY_BOXES = {"DE": (47.3, 55.0, 5.9, 15.0), "IT": (36.6, 47.1, 6.6, 18.5), "ES": (36.0, 43.8, -9.3, 3.3),
                 "FR": (42.3, 51.1, -4.8, 8.2), "NL": (50.75, 53.6, 3.3, 7.2)}
CITIES = {"DE": [("Berlin", 52.52, 13.40), ("Munich", 48.14, 11.58), ("Hamburg", 53.55, 9.99)],
          "IT": [("Milan", 45.46, 9.19), ("Rome", 41.90, 12.50)],
          "ES": [("Madrid", 40.42, -3.70), ("Barcelona", 41.39, 2.17)],
          "FR": [("Paris", 48.86, 2.35), ("Lyon", 45.76, 4.84)],
          "NL": [("Amsterdam", 52.37, 4.90), ("Rotterdam", 51.92, 4.48)]}


def _read(name, ints=(), floats=()):
    with open(SEEDS / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ints:
            r[k] = int(r[k])
        for k in floats:
            r[k] = float(r[k])
    return rows


@lru_cache(maxsize=None)
def _seed(name):
    spec = {"raw_customers": (("customer_id",), ()), "raw_products": (("product_id",), ("unit_price",)),
            "raw_orders": (("order_id", "customer_id", "product_id", "quantity"), ()),
            "raw_payments": (("payment_id", "order_id"), ("amount",)),
            "raw_refunds": (("refund_id", "order_id"), ("amount",)),
            "raw_product_costs": (("product_id",), ("unit_cost",)), "raw_regions": ((), ())}
    ints, floats = spec[name]
    return tuple(tuple(sorted(r.items())) for r in _read(name, ints, floats))


def _rows(name):
    return [dict(r) for r in _seed(name)]


def customers():
    return _rows("raw_customers")


def products():
    return _rows("raw_products")


def orders():
    """Raw orders: one product per order, raw status placed / shipped / returned (dbt maps them to open / fulfilled /
    returned in staging)."""
    return _rows("raw_orders")


def payments():
    return _rows("raw_payments")


def refunds():
    return _rows("raw_refunds")


def _ts(s):
    return datetime.strptime(s, _TS)


def _fmt(t):
    return t.strftime(_TS)


def clickstream():
    """Web events: for every order a session that ends in its checkout, plus browsing sessions that never buy."""
    rnd = random.Random(1001)
    prods = {p["product_id"]: p for p in products()}
    custs = [c["customer_id"] for c in customers()]
    devices = ["desktop", "mobile", "tablet"]
    events, eid = [], 0

    def add(customer, session, kind, ts, payload):
        nonlocal eid
        eid += 1
        events.append({"event_id": eid, "customer_id": customer, "session_id": session, "kind": kind,
                       "event_ts": _fmt(ts), "payload": payload})

    for o in sorted(orders(), key=lambda o: o["order_id"]):
        p, end, session, device = prods[o["product_id"]], _ts(o["order_ts"]), f"s-{o['order_id']:04d}", rnd.choice(devices)
        t = end - timedelta(minutes=rnd.randint(4, 30))
        if rnd.random() < 0.6:
            add(o["customer_id"], session, "search", t, {"query": p["name"] if rnd.random() < 0.7 else p["category"],
                                                         "results": rnd.randint(1, 6), "device": device})
            t += timedelta(minutes=1)
        add(o["customer_id"], session, "page_view", t, {"product_id": p["product_id"], "page": f"/p/{p['name']}",
                                                        "device": device, "referrer": rnd.choice(["search", "ad", "direct"])})
        add(o["customer_id"], session, "add_to_cart", t + timedelta(minutes=2),
            {"product_id": p["product_id"], "quantity": o["quantity"], "unit_price": p["unit_price"]})
        add(o["customer_id"], session, "checkout", end - timedelta(minutes=1),
            {"order_id": o["order_id"], "product_id": p["product_id"], "quantity": o["quantity"],
             "amount": round(o["quantity"] * p["unit_price"], 2)})
    start = min(_ts(o["order_ts"]) for o in orders())
    for i in range(1, 41):                                      # browsing without buying
        c, p = rnd.choice(custs), prods[rnd.choice(sorted(prods))]
        t, session, device = start + timedelta(minutes=rnd.randint(0, 20 * 24 * 60)), f"b-{i:04d}", rnd.choice(devices)
        add(c, session, "page_view", t, {"product_id": p["product_id"], "page": f"/p/{p['name']}", "device": device,
                                         "referrer": rnd.choice(["search", "ad", "direct", "newsletter"])})
        if rnd.random() < 0.35:
            add(c, session, "add_to_cart", t + timedelta(minutes=3), {"product_id": p["product_id"], "quantity": 1,
                                                                      "unit_price": p["unit_price"]})
    return sorted(events, key=lambda e: (e["event_ts"], e["event_id"]))


def deliveries():
    """Delivery point per order: one of the country's hub cities, scattered a few km around it."""
    rnd = random.Random(2002)
    country = {c["customer_id"]: c["country"] for c in customers()}
    out = []
    for o in sorted(orders(), key=lambda o: o["order_id"]):
        cc = country[o["customer_id"]]
        city, lat, lon = rnd.choice(CITIES[cc])
        lat0, lat1, lon0, lon1 = COUNTRY_BOXES[cc]
        out.append({"order_id": o["order_id"], "city": city, "country": cc,
                    "lat": round(min(max(lat + rnd.gauss(0, 0.08), lat0), lat1), 5),
                    "lon": round(min(max(lon + rnd.gauss(0, 0.12), lon0), lon1), 5)})
    return out


def order_events():
    """Orders as they arrived (order_ts order), raw status: the stream iceberg/04 ingests."""
    return sorted(orders(), key=lambda o: (o["order_ts"], o["order_id"]))


# Support story behind the refunds seed: what customers wrote, by product and refund reason.
_TICKETS = {
    ("router", "returned"): "Router drops the Wi-Fi connection every few hours; a reboot helps briefly. Returning it.",
    ("router", "troubleshooting"): "Wi-Fi keeps disconnecting since the firmware update to 2.4.1. Is there a fix?",
    ("ssd", "damaged"): "SSD arrived dead: not detected by the BIOS, the package was crushed.",
    ("ssd", "returned"): "SSD shows as 0 GB after a day of use. Sending it back.",
    ("tablet", "returned"): "Tablet screen flickers at low brightness. Returning it.",
    ("laptop", "returned"): "Laptop fan is very loud under light load. Returning it.",
    ("laptop", "damaged"): "Laptop lid arrived with a dent; it works, partial refund accepted.",
    ("camera", "returned"): "Camera autofocus hunts in low light. Returned.",
    ("headset", "damaged"): "Headset arrived with a cracked headband.",
}


def product_docs():
    """Manuals, reviews and support tickets per product, for RAG. One ticket per refund (topic = refund reason), plus
    troubleshooting tickets for the router's firmware defect before customers gave up and returned it."""
    rnd = random.Random(3003)
    prods = sorted(products(), key=lambda p: p["product_id"])
    order_product = {o["order_id"]: o["product_id"] for o in orders()}
    name = {p["product_id"]: p["name"] for p in prods}
    docs, did = [], 0

    def add(**d):
        nonlocal did
        did += 1
        docs.append({"doc_id": did, "order_id": None, "topic": None, **d})

    for p in prods:
        add(kind="manual", product_id=p["product_id"], title=f"{p['name']} quick start",
            text=f"Northwind Gear {p['name']} ({p['category']}). Setup, warranty (2 years) and troubleshooting. "
                 f"List price {p['unit_price']:.0f} EUR.")
        for i in range(2):
            rating = rnd.choice([3, 4, 5]) if p["name"] not in ("router", "ssd") else rnd.choice([1, 2, 4])
            add(kind="review", product_id=p["product_id"], title=f"{p['name']} review {i + 1}",
                text=f"{rating}/5. " + ("Does the job." if rating >= 4 else "Had problems after a few days."))
    for r in sorted(refunds(), key=lambda r: r["refund_id"]):
        pid = order_product[r["order_id"]]
        text = _TICKETS.get((name[pid], r["reason"]), f"{name[pid]} {r['reason']}: refund requested.")
        add(kind="ticket", product_id=pid, order_id=r["order_id"], topic=r["reason"],
            title=f"Ticket for order {r['order_id']}", text=text)
    for o in sorted(orders(), key=lambda o: o["order_id"]):
        if name[o["product_id"]] == "router" and o["status"] == "returned":
            add(kind="ticket", product_id=o["product_id"], order_id=o["order_id"], topic="troubleshooting",
                title=f"Router support for order {o['order_id']}", text=_TICKETS[("router", "troubleshooting")])
    return docs
