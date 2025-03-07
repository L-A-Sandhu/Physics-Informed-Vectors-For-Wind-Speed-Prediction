import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import mode

# =================================================================
# 1. Enhanced Synthetic Data Generation
# =================================================================
def generate_volatility_data(volatility_level, num_samples=1000):
    """Generate data with clear spatiotemporal volatility differences."""
    np.random.seed(42)
    data = np.zeros((num_samples, 24, 5, 9))  # [samples, time, features, locations]
    
    for sample in range(num_samples):
        # Base signal with location-specific trends
        base = np.sin(np.linspace(0, 4*np.pi, 24))[:, None, None] 
        loc_trend = np.linspace(0, volatility_level, 9)[None, None, :]
        
        # Volatility components
        temporal_noise = volatility_level * np.random.randn(24, 5, 9)
        spatial_diff = volatility_level * np.random.randn(24, 5, 9)
        
        # Feature interactions
        interaction = volatility_level * np.random.randn(24, 5, 9)
        
        data[sample] = base + loc_trend + temporal_noise + spatial_diff + interaction
    
    return data

# Generate distinct classes with 5x difference between low/high
low_vol = generate_volatility_data(0.5, 1000)    # Low
med_vol = generate_volatility_data(0.75, 1000)    # Medium 
high_vol = generate_volatility_data(0.25, 1000)  # High

X = np.concatenate([low_vol, med_vol, high_vol], axis=0)
y_true = np.array([0]*1000 + [1]*1000 + [2]*1000)

# =================================================================
# 2. Robust Feature Engineering
# =================================================================
def compute_discriminative_features(X):
    """Simple but effective volatility features"""
    # Temporal variation
    time_var = np.var(X, axis=1).mean(axis=(1,2))  # Across hours
    
    # Spatial variation
    space_var = np.var(X, axis=3).mean(axis=(1,2))  # Across locations
    
    # Max absolute difference
    max_diff = np.max(np.abs(np.diff(X, axis=1)), axis=(1,2,3))
    
    # Frequency energy
    fft_energy = np.mean(np.abs(np.fft.fft(X, axis=1)[:, 1:3]), axis=(1,2,3))
    
    return np.column_stack([time_var, space_var, max_diff, fft_energy])

X_features = compute_discriminative_features(X)

# =================================================================
# 3. Automated Parameter Optimization
# =================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Parameter grid search
best_ari = -1
best_params = {}
for eps in np.linspace(0.5, 5.0, 20):
    for min_samples in [5, 10, 15]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = db.fit_predict(X_scaled)
        
        if len(np.unique(y_pred)) < 2:
            continue  # Skip single-cluster results
            
        # Calculate ARI for non-noise points
        valid_mask = y_pred != -1
        if valid_mask.sum() == 0:
            continue
            
        ari = adjusted_rand_score(y_true[valid_mask], y_pred[valid_mask])
        if ari > best_ari:
            best_ari = ari
            best_params = {'eps': eps, 'min_samples': min_samples}

print(f"Best parameters: {best_params} | ARI: {best_ari:.3f}")

# =================================================================
# 4. Final Clustering with Validation
# =================================================================
db = DBSCAN(**best_params)
y_pred = db.fit_predict(X_scaled)

# Label mapping with sanity checks
label_map = {}
for cluster in np.unique(y_pred):
    if cluster == -1:
        continue
        
    mask = y_pred == cluster
    if mask.sum() == 0:
        continue
        
    try:
        label_map[cluster] = mode(y_true[mask]).mode.item()
    except:
        label_map[cluster] = -1

y_pred_mapped = np.array([label_map.get(l, -1) for l in y_pred])

# =================================================================
# 5. Performance Evaluation & Visualization
# =================================================================
valid_mask = y_pred_mapped != -1
ari = adjusted_rand_score(y_true[valid_mask], y_pred_mapped[valid_mask])
accuracy = np.mean(y_pred_mapped[valid_mask] == y_true[valid_mask])

print(f"\nAdjusted Rand Index: {ari:.3f}")
print(f"Accuracy (excl. noise): {accuracy:.3f}")
print(f"Noise %: {100*(y_pred == -1).mean():.1f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_mapped))

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12,6))
plt.subplot(121, title="True Labels")
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_true, cmap='viridis', alpha=0.6)
plt.subplot(122, title="Predicted Clusters")
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred_mapped, cmap='viridis', alpha=0.6)
plt.show()