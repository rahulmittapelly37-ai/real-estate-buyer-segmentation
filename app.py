import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

# load data

df = pd.read_csv("clients.csv")

df.head()

df.shape

df.index

df.columns

df.info()

"""Step 1 – Data Cleaning
Tasks performed:

• Handle missing client attributes

• Normalize categorical labels & Remove duplicate client entries
"""

#  Handle Missing Values

df = df.fillna({
    "client_type": "Unknown",
    "gender": "Unknown",
    "country": "Unknown",
    "region": "Unknown",
    "acquisition_purpose": "Unknown",
    "referral_channel": "Unknown",
    "satisfaction_score": df["satisfaction_score"].median()
})

# Normalize Categorical Labels

categorical_cols = ["client_type", "gender", "country", "region",
                    "acquisition_purpose", "referral_channel"]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# Remove Duplicates

df = df.drop_duplicates(subset="client_id")

"""Step 2 – Feature Encoding
Convert categorical fields using:

• One-Hot Encoding

• Label Encoding

• Variables encoded include:
•
client_type
•
region
•
acquisition_purpose
•
referral_channel
•
country

"""

from sklearn.preprocessing import LabelEncoder

# Load dataset

df = pd.read_csv("clients.csv")

# Label Encoding

label_encoded_df = df.copy()
label_encoders = {}

categorical_cols = ["client_type", "region", "acquisition_purpose", "referral_channel", "country"]

for col in categorical_cols:
    le = LabelEncoder()
    label_encoded_df[col] = le.fit_transform(label_encoded_df[col].astype(str))
    label_encoders[col] = le  # store encoder for inverse transform if needed

print("Label Encoded Data:")
print(label_encoded_df.head())

#  One-Hot Encoding

one_hot_encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("One-Hot Encoded Data:")
print(one_hot_encoded_df.head())

"""Step 3 – Feature Scaling
Use StandardScaler or MinMaxScaler to normalize numeric variables such as:

• Age

• Satisfaction score
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

# Load dataset

df = pd.read_csv("clients.csv")

#  Convert date_of_birth to age

def calculate_age(dob):
    for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.now().year - datetime.strptime(str(dob), fmt).year
        except:
            continue
    return np.nan

df["age"] = df["date_of_birth"].apply(calculate_age)

#  Select numeric columns

numeric_cols = ["age", "satisfaction_score"]

#  StandardScaler

scaler_std = StandardScaler()
df_std = df.copy()
df_std[numeric_cols] = scaler_std.fit_transform(df_std[numeric_cols])

print("Standard Scaled Data:")
print(df_std[numeric_cols].head())

"""Step 4 – Clustering Model Selection

Two clustering approaches are used.

• K-Means Clustering Advantages:
● Efficient
● Easy to interpret

• Hierarchical Clustering Advantages:
● Reveals nested cluster relationships
● Helps validate K-means results
"""

#K‑Means Clustering Advantages:
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Assume X is your feature matrix after encoding + scaling

#  Feature Encoding

categorical_cols = ["client_type", "region", "acquisition_purpose", "referral_channel", "country"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

#  Feature Scaling

numeric_cols = ["satisfaction_score", "age"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#  Define Feature Matrix X

X = df[categorical_cols + numeric_cols]

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X)

# Hierarchical Clustering

hc = AgglomerativeClustering(n_clusters=4, linkage="ward")
df["Cluster_HC"] = hc.fit_predict(X)

print(df[["client_id", "Cluster_KMeans", "Cluster_HC"]].head())

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# K-Means cluster distribution

plt.figure(figsize=(8, 5))
sns.countplot(x="Cluster_KMeans", data=df)
plt.title("K-Means Cluster Distribution")
plt.show()

# Hierarchical dendrogram

linked = linkage(X, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

"""Step 5 – Optimal Cluster Selection
Use evaluation methods:

• Elbow Method
Identifies the optimal number of clusters.

• Silhouette Score
Measures clustering quality

🔹 Elbow Method

Plots the inertia (within-cluster sum of squares) against the number of clusters.

The “elbow point” (where the curve bends) suggests the optimal k.
"""

# Assume X is your feature matrix

inertia = []
K = range(2, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

"""🔹 Silhouette Score

Measures how similar each point is to its own cluster compared to other clusters.

Ranges from -1 to 1 → higher values mean better clustering.

"""

from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8,5))
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.show()

"""Cluster Interpretation Dimensions

Investment Purpose

Compare proportions of buyers in each cluster who purchase for Home vs Investment.

Example:

Cluster A → 80% Home buyers (likely first‑time or personal use).

Cluster B → 70% Investment buyers (institutional or high‑net‑worth investors).


Geographic Distribution

Break down clusters by country and region.

Example:

Cluster C → Dominated by California buyers.

Cluster D → More international (Germany, Russia, Belgium).


Loan Behavior

Analyze loan_applied (Yes/No) within each cluster.

Example:

Cluster E → 90% loan‑dependent (first‑time buyers).

Cluster F → 95% self‑financed (luxury or corporate investors).


Customer Demographics

Use age (derived from date_of_birth) and gender.

Example:

Cluster G → Younger demographic (avg age 30–40).

Cluster H → Older demographic (avg age 55+).

Gender balance can also highlight targeting opportunities.
"""

# Summarize clusters

df["Cluster_KMeans"] = kmeans.fit_predict(X)
cluster_summary = df.groupby("Cluster_KMeans").agg({
    "acquisition_purpose": lambda x: (x == "Investment").mean(),
    "loan_applied": lambda x: (x == "Yes").mean(),
    "age": "mean",
    "country": lambda x: x.value_counts().index[0],  # most common country
    "region": lambda x: x.value_counts().index[0]    # most common region
}).reset_index()

print(cluster_summary)
