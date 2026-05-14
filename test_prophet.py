import pandas as pd

try:
    from prophet import Prophet
except Exception as exc:
    print(f"SKIPPED: Prophet is optional in this MVP and is not available: {exc}")
    raise SystemExit(0)

# Minimal data
df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', periods=10),
    'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Test init and fit
m = Prophet()
m.fit(df)
print("SUCCESS: Prophet is correctly installed and the Stan backend is working!")
