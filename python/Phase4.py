import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from Phase1_part2 import run_pipeline

def plot_elbow_and_silhouette(X, k_range):
    """
    Generates Elbow Method and Silhouette Score plots to determine optimal k.
    """
    print("\n Determining Optimal Clusters (k) ")
    
    # Subsample for speed if dataset is huge
    if len(X) > 10000:
        print("   > Subsampling 10k points for Silhouette/Elbow analysis to save time...")
        indices = np.random.choice(len(X), 10000, replace=False)
        X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
    else:
        X_sample = X

    inertia = []
    sil_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_sample)
        inertia.append(kmeans.inertia_)
        
        labels = kmeans.labels_
        score = silhouette_score(X_sample, labels)
        sil_scores.append(score)
        print(f"   k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={score:.4f}")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Elbow Method)', color=color)
    ax1.plot(k_range, inertia, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, sil_scores, marker='s', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Optimal k Analysis: Elbow Method vs Silhouette Score')
    plt.show()
    
    best_k = k_range[np.argmax(sil_scores)]
    print(f"   > Suggested Optimal k (Silhouette Max): {best_k}")
    return best_k

def visualize_clusters_2d(X, labels, title):
    """
    Projects high-dimensional data to 2D using PCA for visualization.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=10, alpha=0.6)
    plt.title(f'{title} (PCA Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()

def run_clustering_loso_stability(X, y_true, groups, k=2):
    """
    Performs Leave-One-Subject-Out (LOSO) validation for Clustering.
    Fits centroids on N-1 subjects, then assigns the held-out subject to those centroids.
    Calculates ARI (Adjusted Rand Index) for the held-out subject to measure generalization.
    """
    print(f"\n Running LOSO Cluster Generalizability Test (k={k}) ")
    
    if groups is None:
        print("   > Warning: No groups provided. Skipping LOSO validation.")
        return

    unique_subs = groups.unique()
    ari_scores = []
    
    logo = LeaveOneGroupOut()
    
    print(f"   > Testing stability across {len(unique_subs)} subjects...")
    
    for train_idx, test_idx in logo.split(X, y_true, groups):
        # Split Data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_test = y_true.iloc[test_idx]
        
        # 1. Fit KMeans on Training Subjects
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3) # Faster n_init for loop
        kmeans.fit(X_train)
        
        # 2. Predict Clusters for Held-Out Subject
        test_labels = kmeans.predict(X_test)
        
        # 3. Check if these clusters match the Ground Truth labels
        score = adjusted_rand_score(y_test, test_labels)
        ari_scores.append(score)
        
    avg_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    print(f"   > Average ARI (Test Sets): {avg_ari:.4f} (+/- {std_ari:.4f})")
    print("   > Interpretation: If Avg ARI is high (>0.5), clusters are stable across new users.")
    print("   > If low, the definition of 'Stress' varies too much between people for unsupervised learning.")

def run_phase_4_clustering(X, y_true, groups=None):
    """
    Executes Phase 4: Clustering Analysis.
    Args:
    - X: The SELECTED features from Phase 1.
    - y_true: Ground Truth labels (0/1).
    - groups: Subject IDs (optional, for stability check).
    """
    print("================ PHASE 4: CLUSTERING ANALYSIS ================")
    print(f"Input Data Shape: {X.shape}")
    print("Note: Using Feature-Selected Data (avoiding Curse of Dimensionality).")
    
    # 1. Determine Optimal k
    k_range = range(2, 7)
    optimal_k = plot_elbow_and_silhouette(X, k_range)
    
    # Force k=2 for direct comparison with Binary Ground Truth
    print(f"\n Running Algorithms with k={2} (Targeting Binary Stress State) ")
    k_clusters = 2
    
    results = []

    #  2. K-MEANS CLUSTERING 
    print("\n1. Training K-Means (Global)...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    sil_km = silhouette_score(X, kmeans_labels, sample_size=10000) 
    db_km = davies_bouldin_score(X, kmeans_labels)
    ari_km = adjusted_rand_score(y_true, kmeans_labels)
    nmi_km = normalized_mutual_info_score(y_true, kmeans_labels)
    
    results.append({
        'Method': 'K-Means',
        'Silhouette': sil_km,
        'Davies-Bouldin': db_km,
        'ARI (Ground Truth)': ari_km,
        'NMI (Ground Truth)': nmi_km
    })
    
    visualize_clusters_2d(X, kmeans_labels, "K-Means Clustering Results")

    #  3. GAUSSIAN MIXTURE MODEL (GMM) 
    print("\n2. Training Gaussian Mixture Model (GMM)...")
    gmm = GaussianMixture(n_components=k_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    
    sil_gmm = silhouette_score(X, gmm_labels, sample_size=10000)
    db_gmm = davies_bouldin_score(X, gmm_labels)
    ari_gmm = adjusted_rand_score(y_true, gmm_labels)
    nmi_gmm = normalized_mutual_info_score(y_true, gmm_labels)
    
    results.append({
        'Method': 'GMM',
        'Silhouette': sil_gmm,
        'Davies-Bouldin': db_gmm,
        'ARI (Ground Truth)': ari_gmm,
        'NMI (Ground Truth)': nmi_gmm
    })
    
    visualize_clusters_2d(X, gmm_labels, "GMM Clustering Results")

    #  4. COMPARISON & ANALYSIS 
    results_df = pd.DataFrame(results)
    print("\n================ CLUSTERING PERFORMANCE COMPARISON ================")
    print(results_df.round(4).to_string(index=False))
    
    #  5. CLUSTER PURITY CHECK 
    print("\n Cluster Purity Analysis (K-Means vs True Labels) ")
    cm = confusion_matrix(y_true, kmeans_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix: Unsupervised Clusters vs True Labels')
    plt.xlabel('Cluster ID')
    plt.ylabel('True Label')
    plt.show()
    
    #  6. LOSO STABILITY CHECK (New Addition) 
    if groups is not None:
        run_clustering_loso_stability(X, y_true, groups, k=k_clusters)

if __name__ == "__main__":
    X_final, y_reg, y_class, groups = run_pipeline()
    run_phase_4_clustering(X_final, y_class, groups)