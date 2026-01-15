import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import cond, svd

def run_phase_1_eda(master_df):
    """
    Phase 1: The Cleanup & Sanity Check.
    We take the messy raw data, clean it up, normalize it so the math works, 
    and visualize it to see if we're actually going to get good results.
    """
    print("="*50)
    print("PHASE I: FEATURE ENGINEERING & EDA")
    print("="*50)

    #  1. SETUP 
    # Defining what we want to predict (Label = Stress vs Not Stress)
    targets = ['label', 'valence', 'arousal']
    # Stuff that helps us identify rows but isn't a "feature" for the model
    meta = ['time', 'subject']
    
    # Grab every column that isn't a target or an ID
    feature_cols = [c for c in master_df.columns if c not in targets + meta]
    
    print(f"Total Features identified: {len(feature_cols)}")
    print(f"Target Variable (Class): label (0=Baseline/Fun, 1=Stress)")
    print(f"Target Variables (Reg): valence, arousal")

    #  2. CLEANING 
    print("\n 2. Data Cleaning & Preprocessing ")
    
    # Check if we accidentally copied rows (happens sometimes with merging)
    dupes = master_df.duplicated().sum()
    print(f"Duplicated rows found: {dupes}")
    if dupes > 0:
        master_df = master_df.drop_duplicates()
        print("Duplicates removed.")
        
    # Check for empty values. If the sensor disconnected, we just drop the row.
    initial_shape = master_df.shape
    df_clean = master_df.dropna()
    print(f"NaNs handling: Dropped {initial_shape[0] - df_clean.shape[0]} rows containing NaNs.")
    
    #  3. NORMALIZATION (The Important Part) 
    print("\n 3. Variable Transformation (Subject-Specific Standardization) ")
    
    df_scaled = df_clean.copy()
    scaler = StandardScaler()
    subjects = df_clean['subject'].unique()
    
    # We loop through subjects and scale them individually.
    # Because Person A might have a resting HR of 60, and Person B has 80.
    # If we scaled them together, Person B would look "stressed" all the time.
    # We need Z-scores relative to their baseline.
    for subj in subjects:
        mask = df_clean['subject'] == subj
        if mask.sum() > 0:
            df_scaled.loc[mask, feature_cols] = scaler.fit_transform(df_clean.loc[mask, feature_cols])
        
    print("Data standardized (Z-score) per subject.")

    #  4. OUTLIER REMOVAL 
    print("\n 4. Anomaly Detection (Isolation Forest) ")
    # Using Isolation Forest to find data points that look "weird" compared to the rest.
    # Contamination=0.05 means we expect about 5% of the data to be garbage/artifacts.
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    
    X_detect = df_scaled[feature_cols]
    outliers = iso.fit_predict(X_detect) # -1 means it's an outlier, 1 is normal
    
    mask_inliers = outliers != -1
    df_final = df_scaled[mask_inliers].copy()
    
    # We keep the 'subject' column for now so we can do Leave-One-Subject-Out (LOSO) later
    
    print(f"Outliers detected and removed: {np.sum(outliers == -1)}")
    print(f"Final Data Shape: {df_final.shape}")

    # Setup X and y just for the charts/stats below
    eda_targets = targets
    eda_feature_cols = [c for c in df_final.columns if c not in eda_targets + meta]
    
    X = df_final[eda_feature_cols]
    y = df_final['label'].astype(int) 

    #  5. MATH CHECKS (Checking for Redundancy) 
    print("\n 5. Dimensionality Reduction & Feature Selection (EDA Stats) ")
    
    # Condition Number: If this is huge, our matrix is unstable (features are too correlated).
    try:
        c_num = cond(X)
        print(f"Condition Number: {c_num:.2e}")
        print("Observation: High condition number implies strong multicollinearity.")
    except Exception as e:
        print(f"Condition Number Check Skipped: {e}")

    # SVD: Just checking the singular values to see how fast they drop off.
    try:
        U, s, Vt = svd(X, full_matrices=False)
        print(f"Top 5 Singular Values: {s[:5]}")
    except Exception as e:
        print(f"SVD Skipped: {e}")
    
    # VIF (Variance Inflation Factor):
    # This specifically hunts for features that are copies of other features.
    print("Calculating VIF (Top 10)...")
    # Taking a small sample because VIF is slow on huge datasets
    X_sample = X.sample(n=min(5000, len(X)), random_state=42)
    X_sample = X_sample.select_dtypes(include=[np.number])
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_sample.columns
    vif_data["VIF"] = [variance_inflation_factor(X_sample.values, i) for i in range(X_sample.shape[1])]
    print(vif_data.sort_values('VIF', ascending=False).head(10))

    # PCA: Squashing the data to see how many dimensions we actually need.
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)
    print(f"PCA: Reduced features from {X.shape[1]} to {X_pca.shape[1]} to explain 95% variance.")
    
    # LDA: Trying to see if the classes separate well linearly.
    try:
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X, y)
        print(f"LDA: Explained Variance Ratio {lda.explained_variance_ratio_}")
    except Exception as e:
        print(f"LDA Skipped: {e}")
    
    # Random Forest Importance:
    # Asking a basic Random Forest "Which columns actually helped you make decisions?"
    print("Calculating Random Forest Feature Importance for EDA...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Features (Random Forest)")
    plt.show()
    
    #  6. HEATMAPS 
    print("\n 6. Visualizations (Heatmaps) ")
    
    # Covariance Matrix: How do features vary together?
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.cov(), cmap='viridis', xticklabels=True)
    plt.title("Sample Covariance Matrix")
    plt.show()
    
    # Correlation Matrix: Are features positively/negatively correlated to the Target?
    plt.figure(figsize=(12, 10))
    X_corr = X.copy()
    X_corr['Target_Label'] = y
    sns.heatmap(X_corr.corr(), cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
    plt.title("Pearson Correlation Matrix")
    plt.show()

    #  7. CHECKING BALANCE 
    print("\n 7. Class Imbalance Check ")
    # If we have 90% non-stress and 10% stress, our model might just guess "non-stress" every time.
    print("Original Class Distribution:")
    print(y.value_counts(normalize=True))
    
    print("\nPHASE I COMPLETE.")
    return df_final

def post_process_features(df):
    """
    Phase 2: The Cut.
    We actually drop the bad features here based on the stats we calculated above.
    """
    print("\n================ POST-PROCESSING: FEATURE SELECTION ================")
    
    targets = ['label', 'valence', 'arousal']
    
    # Need to keep track of which subject is which for cross-validation later
    if 'subject' in df.columns:
        groups = df['subject']
        # Drop the meta stuff from the training set
        X_raw = df.drop(columns=targets + ['subject', 'time'], errors='ignore')
    else:
        raise ValueError("Subject column missing! Cannot perform LOSO validation.")

    y_class = df['label']
    y_reg = df[['valence', 'arousal']]
    
    print(f"Starting Features: {X_raw.shape[1]}")

    #  STEP A: VIF FILTER 
    # We iterate here. If we drop the worst feature, the VIF scores for everyone else change.
    # So we drop one, recalculate, drop the next one, until everyone behaves.
    print("Running Iterative VIF Filtering...")
    X = X_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    
    vif_threshold = 10 # Standard cutoff for "way too correlated"
    dropped_vif = []
    X_vif_sample = X.sample(n=min(2000, len(X)), random_state=42)
    
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif_sample.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(X_vif_sample.values, i) for i in range(X_vif_sample.shape[1])]
        except Exception as e:
            print(f"VIF Error: {e}")
            break
        
        max_vif = vif_data['VIF'].max()
        if max_vif > vif_threshold:
            max_vif_feature = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
            X = X.drop(columns=[max_vif_feature])
            X_vif_sample = X_vif_sample.drop(columns=[max_vif_feature])
            dropped_vif.append(max_vif_feature)
        else:
            break
            
    print(f"VIF Filter Complete. Dropped {len(dropped_vif)} columns.")

    #  STEP B: RANDOM FOREST SELECTION 
    # Now that we removed the redundant stuff, we let a Random Forest pick the most predictive features.
    print("Running Random Forest Selector...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X, y_class)
    
    # Sklearn's SelectFromModel automatically picks features above the mean importance
    selector = SelectFromModel(rf_selector, prefit=True)
    selected_cols = X.columns[selector.get_support()]
    X_final = X[selected_cols]
    
    print(f"RF Filter Complete. Selected Top {len(selected_cols)} Features.")
    print(f"Final Feature List: {list(selected_cols)}")

    print(f"Data Prepared. Features Shape: {X_final.shape}")
    
    return X_final, y_reg, y_class, groups

def run_pipeline():
    master_df = pd.read_csv("wesad_features.csv")
    cleaned_df = run_phase_1_eda(master_df)
    return post_process_features(cleaned_df)