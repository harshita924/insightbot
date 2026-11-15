"""Generates a SYNTHETIC Superstore-style retail dataset (no real/confidential data).
Run:  python make_data.py   ->  data/superstore_synthetic.csv
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
SUBS = {  # sub_category: (category, price_lo, price_hi, base_margin)
    "Bookcases": ("Furniture", 150, 600, -0.02), "Chairs": ("Furniture", 100, 700, 0.10),
    "Furnishings": ("Furniture", 15, 120, 0.15), "Tables": ("Furniture", 200, 1100, -0.08),
    "Binders": ("Office Supplies", 5, 60, 0.30), "Paper": ("Office Supplies", 5, 50, 0.38),
    "Storage": ("Office Supplies", 20, 250, 0.10), "Art": ("Office Supplies", 4, 40, 0.28),
    "Labels": ("Office Supplies", 3, 30, 0.42), "Appliances": ("Office Supplies", 30, 500, 0.17),
    "Phones": ("Technology", 100, 900, 0.15), "Accessories": ("Technology", 10, 200, 0.25),
    "Machines": ("Technology", 200, 2500, 0.05), "Copiers": ("Technology", 600, 3500, 0.32),
}
sub_names = list(SUBS)
sub_w = np.array([3, 6, 6, 3, 14, 12, 7, 8, 5, 3, 7, 9, 2, 1], float); sub_w /= sub_w.sum()
regions = ["West", "East", "Central", "South"]; reg_w = [0.32, 0.28, 0.22, 0.18]
segs = ["Consumer", "Corporate", "Home Office"]; seg_w = [0.52, 0.30, 0.18]

days = pd.date_range("2023-01-01", "2026-09-30")
growth = 1 + 0.35 * (np.arange(len(days)) / len(days))
season = np.where(days.month.isin([11, 12]), 1.6, np.where(days.month.isin([9, 3]), 1.2, 1.0))
w = growth * season; w /= w.sum()

N_ORD = 5200
order_dates = rng.choice(days, size=N_ORD, p=w)
order_region = rng.choice(regions, size=N_ORD, p=reg_w)
order_seg = rng.choice(segs, size=N_ORD, p=seg_w)
rows = []
for i in range(N_ORD):
    d = pd.Timestamp(order_dates[i]); reg = order_region[i]
    in_q3_26 = (d >= pd.Timestamp("2026-07-01"))
    # Planted story: West ran heavy discounting in Q3 2026 and lost volume
    if reg == "West" and in_q3_26 and rng.random() < 0.18:
        continue  # fewer West orders -> lower sales
    for _ in range(int(rng.integers(1, 4))):
        sub = rng.choice(sub_names, p=sub_w); cat, lo, hi, bm = SUBS[sub]
        qty = int(rng.integers(1, 8))
        price = rng.uniform(lo, hi)
        dp = [0.35, 0.25, 0.20, 0.12, 0.08]
        if reg == "West" and in_q3_26: dp = [0.10, 0.20, 0.25, 0.25, 0.20]
        disc = float(rng.choice([0, 0.1, 0.2, 0.3, 0.4], p=dp))
        if cat == "Furniture" and in_q3_26: bm -= 0.03
        sales = qty * price * (1 - disc)
        profit = sales * (bm - 0.45 * disc) + rng.normal(0, 0.02 * sales)
        rows.append((f"ORD-{d.year}-{i+1:05d}", d.strftime("%Y-%m-%d"), reg, order_seg[i], cat, sub,
                     qty, disc, round(sales, 2), round(profit, 2)))
df = pd.DataFrame(rows, columns=["order_id", "order_date", "region", "segment", "category",
                                 "sub_category", "quantity", "discount", "sales", "profit"])
df.to_csv("data/superstore_synthetic.csv", index=False)
print(df.shape, df.order_date.min(), df.order_date.max())
