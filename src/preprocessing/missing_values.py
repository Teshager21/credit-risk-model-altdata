def report_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print("🧼 Missing Values:")
        print(missing)
    else:
        print("✅ No missing values found.")
