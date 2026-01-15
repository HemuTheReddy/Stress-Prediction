import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from Phase1_part2 import run_pipeline


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def plot_knn_elbow(X, y, groups):
    """
    Finds optimum K using the Elbow Method with Group Split to avoid leakage.
    """
    print("\n KNN Optimization: Running Elbow Method (Group Split) ")
    error_rate = []
    k_range = range(1, 31)
    
    # Subsample groups for elbow optimization if dataset is massive    
    # Use GroupKFold for a quick safe split
    gkf = GroupKFold(n_splits=3)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    
    # If training set is massive (>20k), subset it for the elbow plot
    if len(train_idx) > 20000:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(train_idx, 20000, replace=False)
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('KNN Elbow Method (Error Rate vs K) - Group Validation')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.show()
    
    best_k = k_range[np.argmin(error_rate)]
    print(f"Elbow Method determined optimal K = {best_k}")
    return best_k

def analyze_dt_pruning(X, y, groups):
    """
    Analyzes Decision Tree Cost Complexity Pruning (CCP) path using Group Split.
    """
    print("\n Decision Tree: Cost Complexity Pruning Analysis (Group Split) ")
    
    gkf = GroupKFold(n_splits=3)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    
    # Subsample if massive
    if len(train_idx) > 20000:
        rng = np.random.RandomState(42)
        train_idx = rng.choice(train_idx, 20000, replace=False)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    plt.xlabel("effective alpha")
    plt.ylabel("total impurity of leaves")
    plt.title("Total Impurity vs effective alpha")
    plt.grid(True)
    plt.show()
    
    return ccp_alphas[::max(1, len(ccp_alphas)//10)]

def run_phase_3_classification_loso(X, y, groups):
    """
    Main Phase 3 Pipeline with LOSO (Leave-One-Subject-Out) Validation.
    """
    print("================ PHASE 3: CLASSIFICATION ANALYSIS (LOSO) ================")
    
    if groups is None:
        raise ValueError("Groups (Subject IDs) are required for LOSO validation.")
        
    cv_outer = LeaveOneGroupOut()
    cv_inner = GroupKFold(n_splits=3)
    
    # 1. PRELIMINARY OPTIMIZATIONS 
    best_k = plot_knn_elbow(X, y, groups)
    dt_alphas = analyze_dt_pruning(X, y, groups)

    # 2. DEFINE CLASSIFIERS & HYPERPARAMETERS 
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svr', SVC(probability=True, random_state=42))
    ]
    
    models_config = {
        'LDA': {
            'model': LinearDiscriminantAnalysis(),
            'params': {'classifier__solver': ['svd', 'lsqr']}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__splitter': ['best', 'random'],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__max_features': ['sqrt', 'log2', None],
                'classifier__ccp_alpha': dt_alphas
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [best_k, best_k+2, max(1, best_k-2)], 
                'classifier__weights': ['uniform', 'distance']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'classifier__kernel': ['linear', 'rbf'], 
                'classifier__C': [0.1, 1],
                'classifier__gamma': ['scale']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}
        },
        'Random Forest (Bagging)': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10],
                'classifier__criterion': ['gini']
            }
        },
        'Gradient Boosting (Boosting)': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50],
                'classifier__learning_rate': [0.1]
            }
        },
        'Stacking Classifier': {
            'model': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()),
            'params': {}
        },
        'Neural Network (MLP)': {
            'model': MLPClassifier(max_iter=500, random_state=42),
            'params': {
                'classifier__hidden_layer_sizes': [(50,), (100,)],
                'classifier__activation': ['relu'],
                'classifier__alpha': [0.0001]
            }
        }
    }

    results_summary = []
    roc_data = {}
    
    # List of expensive models to subsample
    expensive_models = ['SVM', 'Stacking Classifier', 'Neural Network (MLP)']

    # 3. TRAINING LOOP 
    for name, config in models_config.items():
        print(f"\nProcessing: {name}...")
        
        # COMPUTATIONAL OPTIMIZATION: Subsample for Expensive Models 
        X_used, y_used, groups_used = X, y, groups
        if name in expensive_models and len(X) > 20000:
             print(f"   > Optimization: Subsampling 20k samples for {name}...")
             # Use numpy for random index generation to keep groups aligned
             rng = np.random.RandomState(42)
             indices = rng.choice(len(X), 20000, replace=False)
             X_used = X.iloc[indices]
             y_used = y.iloc[indices]
             groups_used = groups.iloc[indices]

        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', config['model'])
        ])
        
        # Grid Search (Inner Loop)
        if config['params']:
            print(f"   > Performing Grid Search (GroupKFold)...")
            gs = GridSearchCV(pipeline, config['params'], cv=cv_inner, scoring='f1', n_jobs=-1)
            gs.fit(X_used, y_used, groups=groups_used)
            best_model = gs.best_estimator_
            print(f"   > Best Params: {gs.best_params_}")
        else:
            best_model = pipeline

        # Cross Validation (Outer Loop - LOSO)
        print(f"   > Running Leave-One-Subject-Out CV...")
        y_pred = cross_val_predict(best_model, X_used, y_used, cv=cv_outer, groups=groups_used, n_jobs=-1)
        
        try:
            y_proba = cross_val_predict(best_model, X_used, y_used, cv=cv_outer, groups=groups_used, method='predict_proba', n_jobs=-1)[:, 1]
        except:
            if hasattr(best_model, "decision_function"):
                 y_proba = cross_val_predict(best_model, X_used, y_used, cv=cv_outer, groups=groups_used, method='decision_function', n_jobs=-1)
            else:
                y_proba = np.zeros_like(y_pred)

        # 4. METRICS 
        cm = confusion_matrix(y_used, y_pred)
        acc = accuracy_score(y_used, y_pred)
        prec = precision_score(y_used, y_pred, zero_division=0)
        rec = recall_score(y_used, y_pred, zero_division=0)
        spec = calculate_specificity(y_used, y_pred)
        f1 = f1_score(y_used, y_pred, zero_division=0)
        
        if len(np.unique(y_used)) > 1:
            fpr, tpr, _ = roc_curve(y_used, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = [0], [0], 0.5
            
        roc_data[name] = (fpr, tpr, roc_auc)
        
        results_summary.append({
            'Classifier': name,
            'Accuracy': acc,
            'Precision': prec,
            'Sensitivity (Recall)': rec,
            'Specificity': spec,
            'F-Score': f1,
            'AUC': roc_auc
        })
        
        print(f"   > Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {roc_auc:.3f}")
        
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    #  5. VISUALIZATION 
    print("\n Comparative ROC Curves ")
    plt.figure(figsize=(12, 8))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve (LOSO Validation)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values(by='F-Score', ascending=False)
    
    print("\n================ FINAL CLASSIFIER COMPARISON (LOSO) ================")
    print(results_df.round(4).to_string(index=False))
    
    best_model_name = results_df.iloc[0]['Classifier']
    print(f"\n>>> RECOMMENDATION: The best classifier is '{best_model_name}' based on F-Score.")
    
    plt.figure(figsize=(14, 6))
    melted_df = results_df.melt(id_vars='Classifier', value_vars=['Accuracy', 'Sensitivity (Recall)', 'Specificity', 'F-Score'])
    sns.barplot(x='Classifier', y='value', hue='variable', data=melted_df)
    plt.title("Performance Metrics Comparison (LOSO)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_final, y_reg, y_class, groups = run_pipeline()
    run_phase_3_classification_loso(X_final, y_class, groups)