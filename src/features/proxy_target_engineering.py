# src/features/proxy_target_engineering.py

import pandas as pd

# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ProxyTargetEngineer:
    """
    Engineer a proxy target variable for credit risk using RFM analysis
    and KMeans clustering.
    """

    def __init__(self, snapshot_date, n_clusters=3, random_state=42):
        """
        Parameters
        ----------
        snapshot_date : str or pd.Timestamp
            The cutoff date used to compute recency.

        n_clusters : int, default=3
            Number of KMeans clusters.

        random_state : int, default=42
            Random state for reproducibility.
        """
        self.snapshot_date = pd.Timestamp(snapshot_date).tz_localize("UTC")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.cluster_labels_ = None
        self.risk_cluster_ = None
        self.rfm_features_ = None

    def calculate_rfm(
        self,
        df,
        customer_id_col="CustomerId",
        date_col="TransactionStartTime",
        amount_col="TransactionAmount",
    ):
        """
        Compute RFM metrics from transaction data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data.

        Returns
        -------
        rfm : pd.DataFrame
            DataFrame with columns:
                CustomerId, Recency, Frequency, Monetary
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Monetary
        monetary = (
            df.groupby(customer_id_col)[amount_col].sum().reset_index(name="Monetary")
        )

        # Frequency
        frequency = (
            df.groupby(customer_id_col)[date_col].count().reset_index(name="Frequency")
        )

        # Recency
        recency = df.groupby(customer_id_col)[date_col].max().reset_index()
        recency["Recency"] = (self.snapshot_date - recency[date_col]).dt.days
        recency = recency[[customer_id_col, "Recency"]]

        # Merge
        rfm = monetary.merge(frequency, on=customer_id_col)
        rfm = rfm.merge(recency, on=customer_id_col)

        self.rfm_features_ = ["Recency", "Frequency", "Monetary"]

        return rfm

    def cluster_customers(self, rfm_df, customer_id_col="CustomerId"):
        """
        Cluster customers into segments based on RFM metrics.

        Parameters
        ----------
        rfm_df : pd.DataFrame
            DataFrame with RFM metrics.

        Returns
        -------
        rfm_df_with_cluster : pd.DataFrame
            RFM DataFrame with an added 'Cluster' column.
        """
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_df[self.rfm_features_])

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(rfm_scaled)

        self.kmeans_model = kmeans
        self.cluster_labels_ = clusters

        rfm_df_with_cluster = rfm_df.copy()
        rfm_df_with_cluster["Cluster"] = clusters

        return rfm_df_with_cluster

    def identify_high_risk_cluster(
        self, rfm_clustered_df, customer_id_col="CustomerId"
    ):
        """
        Determine which cluster represents high-risk customers.

        Parameters
        ----------
        rfm_clustered_df : pd.DataFrame

        Returns
        -------
        high_risk_cluster : int
            The label of the high-risk cluster.
        """
        # Low frequency and low monetary usually signal disengaged customers
        cluster_summary = rfm_clustered_df.groupby("Cluster")[
            ["Frequency", "Monetary"]
        ].mean()
        # The cluster with the lowest frequency and monetary average
        cluster_summary["Freq_Mon"] = (
            cluster_summary["Frequency"] + cluster_summary["Monetary"]
        )
        high_risk_cluster = cluster_summary["Freq_Mon"].idxmin()

        self.risk_cluster_ = high_risk_cluster
        return high_risk_cluster

    def assign_high_risk_label(self, rfm_clustered_df, customer_id_col="CustomerId"):
        """
        Assign binary is_high_risk label.

        Parameters
        ----------
        rfm_clustered_df : pd.DataFrame

        Returns
        -------
        high_risk_labels : pd.DataFrame
            DataFrame with CustomerId and is_high_risk.
        """
        high_risk_cluster = self.risk_cluster_
        labels = rfm_clustered_df[[customer_id_col, "Cluster"]].copy()
        labels["is_high_risk"] = (labels["Cluster"] == high_risk_cluster).astype(int)
        labels = labels.drop(columns="Cluster")
        return labels

    def transform(
        self,
        transactions_df,
        customer_id_col="CustomerId",
        date_col="TransactionStartTime",
        amount_col="TransactionAmount",
    ):
        """
        Full pipeline to generate is_high_risk label and merge into main dataset.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            The transaction history data.

        Returns
        -------
        high_risk_labels : pd.DataFrame
            DataFrame with CustomerId and is_high_risk column.
        """
        rfm = self.calculate_rfm(transactions_df, customer_id_col, date_col, amount_col)
        clustered = self.cluster_customers(rfm, customer_id_col)
        self.identify_high_risk_cluster(clustered, customer_id_col)
        labels = self.assign_high_risk_label(clustered, customer_id_col)
        return labels
