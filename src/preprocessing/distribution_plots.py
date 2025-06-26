import matplotlib.pyplot as plt

# import seaborn as sns


def plot_numerical_distributions(df):
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols].hist(figsize=(14, 10), bins=30)
    plt.suptitle("Numerical Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df):
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        plt.figure(figsize=(10, 4))
        df[col].value_counts(normalize=True).head(10).plot(kind="bar")
        plt.title(f"Top categories in {col}")
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.show()
