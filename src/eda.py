def profile(df):
    print("Shape:", df.shape)
    print(df.isnull().mean().sort_values(ascending=False).head(20))
    for c in df.columns:
        print(f"{c}: {df[c].nunique()} unique | {df[c].dropna().unique()[:3]}")
    # Hour-1 questions: exact metric? target skew? imbalance?
    #                    repeating IDs? text cols? train/test shift?
