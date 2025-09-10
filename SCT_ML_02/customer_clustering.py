import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Preview dataset
print("Dataset preview:")
print(data.head())

# Select 3 features for clustering
X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Find optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot elbow method
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method (3D features)")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans with optimal clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to dataset
data["Cluster"] = y_kmeans
data.to_csv("clustered_customers_3d.csv", index=False)
print("✅ Clustered customer data saved to clustered_customers_3d.csv")

# 3D Visualization of Clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X["Age"], X["Annual Income (k$)"], X["Spending Score (1-100)"],
    c=y_kmeans, cmap="rainbow", s=50
)

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.set_title("3D Customer Segments")

plt.show()
