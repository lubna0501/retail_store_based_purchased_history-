import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset containing purchase history from the C drive
# Replace 'C:/path/to/purchase_data.csv' with the actual path to your dataset
file_path = r"C:\Users\lubna\Downloads\archive (12)\Mall_Customers.csv"

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

data = data.drop(columns=['CustomerID'])

# Convert categorical variable 'Gender' into numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

# Separate features (X) and target (y)
X = data.drop(columns=['Spending Score (1-100)'])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Display the first few rows of the dataset
print(data.head())

# Extract features (purchase behavior) for clustering
#X = data.iloc[:, 1:]  # Exclude customer ID column

# Standardize the features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
# Sum of squared distances of samples to their closest cluster center
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Based on the Elbow Method, select the optimal number of clusters (e.g., where the curve starts to flatten)
n_clusters = 3  # Update with the chosen number of clusters

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Display the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
print("Cluster Centers:")
print(cluster_centers_df)

# Visualize the clusters (for 2D features only)
# Replace 'feature1' and 'feature2' with the features you want to visualize
plt.figure(figsize=(10, 6))
plt.scatter(X['Age'], X['Annual Income (k$)'], c=clusters, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Clustering of Customers')
plt.show()

# Analyze the characteristics of each cluster
cluster_analysis = data.groupby('Cluster').mean()
print("Cluster Analysis:")
print(cluster_analysis)
