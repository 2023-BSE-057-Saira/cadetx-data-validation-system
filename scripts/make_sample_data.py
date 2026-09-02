import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

invoice_no = [f"5{rng.integers(30000, 39999)}" for _ in range(n)]
stock_code = [f"{rng.integers(10000, 99999)}" for _ in range(n)]
descriptions = [
    "WHITE HANGING HEART T-LIGHT HOLDER", "RED WOOLLY HOTTIE WHITE HEART.",
    "SET 7 BABUSHKA NESTING BOXES", None, "GLASS STAR FROSTED T-LIGHT HOLDER",
    "assorted colour bird ornament", "POPPY'S PLAYHOUSE BEDROOM", "n/a",
]
quantity = rng.integers(-10, 50, n)
unit_price = np.round(rng.uniform(-1, 50, n), 2)
customer_id = rng.choice([np.nan, *rng.integers(12000, 18000, 200)], n)
country = rng.choice(
    ["United Kingdom", "France", "Germany", "eire", "  Spain", "United Kingdom", None], n,
)
invoice_date = pd.date_range("2010-12-01", "2011-12-09", periods=n)
email = rng.choice(
    ["contact@shop.com", "not-an-email", None, "user@@bad.com", "hello@retail.co.uk"], n
)
phone = rng.choice(["+44 20 7946 0958", "07123456789", "not-a-phone", None, "12345"], n)

df = pd.DataFrame({
    "InvoiceNo": invoice_no,
    "StockCode": stock_code,
    "Description": rng.choice(descriptions, n),
    "Quantity": quantity,
    "InvoiceDate": invoice_date,
    "UnitPrice": unit_price,
    "CustomerID": customer_id,
    "Country": country,
    "ContactEmail": email,
    "ContactPhone": phone,
})

df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)
df["Quantity"] = df["Quantity"].astype(object)
df.loc[df.sample(10, random_state=2).index, "Quantity"] = "unknown"

df.to_csv("/home/claude/cadetx-project/data/raw/sample_retail.csv", index=False)
print(f"Wrote {len(df)} rows")
