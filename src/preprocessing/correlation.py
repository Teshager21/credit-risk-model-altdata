import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    num_cols = df.select_dtypes(include="number").columns
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
