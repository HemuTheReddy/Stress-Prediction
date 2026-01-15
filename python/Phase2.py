from Phase1_part2 import run_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score


def stepwise_selection_bidirectional(X, y, verbose=False):
    """    
    Standard models often overfit if we throw 50 features at them.
    This function tries to find the perfect subset by:
    1. Forward: Adding the best remaining feature.
    2. Backward: Checking if any existing feature is now useless and deleting it.
    3. Repeat until the Adjusted R-squared stops going up.
    """
    initial_features = []
    included = list(initial_features)
    
    current_score = -np.inf
    
    if verbose:
        print("\n Stepwise Regression (Target: Maximize Adj. R2) ")
        
    while True:
        changed = False
        
        #  FORWARD STEP (Try adding stuff) 
        excluded = list(set(X.columns) - set(included))
        best_new_score = current_score
        best_candidate = None
        
        for candidate in excluded:
            # Create a temp model with the new candidate
            features_to_test = included + [candidate]
            X_const = sm.add_constant(X[features_to_test])
            try:
                model = sm.OLS(y, X_const).fit()
                score = model.rsquared_adj
                # If this candidate makes the model better, track it
                if score > best_new_score:
                    best_new_score = score
                    best_candidate = candidate
            except:
                continue
        
        # If we found a winner, officially add it to our list
        if best_candidate is not None:
            included.append(best_candidate)
            current_score = best_new_score
            changed = True
            if verbose:
                print(f"   [+] Added: {best_candidate:<25} Adj.R2: {current_score:.4f}")

        #  BACKWARD STEP (Try pruning stuff) 
        # Sometimes adding Feature C makes Feature A redundant. We need to check for that.
        if len(included) > 0:
            best_score_after_removal = current_score
            candidate_to_remove = None
            
            for candidate in included:
                features_remaining = list(set(included) - {candidate})
                
                # If we remove the last feature, score is 0
                if len(features_remaining) == 0:
                    score = 0
                else:
                    X_const = sm.add_constant(X[features_remaining])
                    try:
                        model = sm.OLS(y, X_const).fit()
                        score = model.rsquared_adj
                    except:
                        score = -np.inf
                
                # If removing the feature actually improved (or maintained) the score, cut it
                if score > best_score_after_removal:
                    best_score_after_removal = score
                    candidate_to_remove = candidate
            
            if candidate_to_remove is not None:
                included.remove(candidate_to_remove)
                current_score = best_score_after_removal
                changed = True
                if verbose:
                    print(f"   [-] Removed: {candidate_to_remove:<23} Adj.R2: {current_score:.4f}")
        
        # If we went through a whole loop and didn't add OR remove anything, we're done.
        if not changed:
            break

    return included

# 
def run_phase_2_regression(X, y_reg, groups):
    """
    The Main Regression Loop.
    
    We are predicting continuous values:
    1. Valence (How negative/positive the emotion is)
    2. Arousal (How intense/excited the emotion is)
    
    We use Leave-One-Subject-Out (LOSO) validation.
    If we train on Subject 2 and Test on Subject 2, the model cheats by memorizing 
    their specific heart rate baseline. We must Train on S2-S10, and Test on S11.
    """
    
    # Keeping track of predictions for the final "Map of Emotions" plot
    circumplex_data = {
        'valence': {'true': [], 'pred': []},
        'arousal': {'true': [], 'pred': []}
    }
    
    targets = ['valence', 'arousal']
    
    for target_col in targets:
        print(f"\n{'='*60}")
        print(f"PHASE 2: REGRESSION ANALYSIS FOR TARGET: {target_col.upper()}")
        print(f"{'='*60}")
        
        unique_subs = groups.unique()
        results_list = []
        
        # Arrays for the aggregate scatter plot
        curr_target_true = []
        curr_target_pred = []
        
        print(f"Starting Leave-One-Subject-Out (LOSO) Validation for {len(unique_subs)} subjects...")

        #  THE LOSO LOOP 
        for subj in unique_subs:
            # Create boolean masks to split the data
            train_mask = groups != subj # Train on everyone else
            test_mask = groups == subj  # Test on this specific person
            
            # Split Data
            X_train_full = X[train_mask]
            y_train = y_reg.loc[train_mask, target_col]
            X_test_full = X[test_mask]
            y_test = y_reg.loc[test_mask, target_col]
            
            # A. Feature Selection (Only look at Train data!)
            # If we looked at Test data here, it would be "Data Leakage" -> bad science.
            selected_features = stepwise_selection_bidirectional(X_train_full, y_train, verbose=False)
            
            # Fallback if stepwise deletes everything (rare, but happens)
            if len(selected_features) == 0:
                print(f"Warning: No features selected for Subject {subj}. Using all.")
                selected_features = list(X.columns)

            # B. Train the Model (OLS = Ordinary Least Squares)
            X_train_opt = sm.add_constant(X_train_full[selected_features])
            X_test_opt = sm.add_constant(X_test_full[selected_features])
            
            model = sm.OLS(y_train, X_train_opt).fit()
            
            # C. Predict on the "Unknown" Subject
            y_pred = model.predict(X_test_opt)
            
            # D. Calculate how well we did
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred) 
            
            results_list.append({
                'Subject': subj,
                'R-squared (Test)': r2,
                'MSE (Test)': mse,
                'Adj. R-squared (Train)': model.rsquared_adj, # How well it fit the training data
                'AIC': model.aic,
                'BIC': model.bic,
                'Num_Features': len(selected_features)
            })
            
            # Store data for the big plots at the end
            curr_target_true.extend(y_test)
            curr_target_pred.extend(y_pred)
            
            circumplex_data[target_col]['true'].extend(y_test)
            circumplex_data[target_col]['pred'].extend(y_pred)
            
            #  Single Subject Deep Dive 
            # We only print the full stats for the FIRST subject so the console doesn't explode.
            if subj == unique_subs[0]:
                print(f"\n Detailed Analysis for Example Subject: {subj} ({target_col}) ")
                print(model.summary())
                print("\n Confidence Intervals (95%) ")
                print(model.conf_int())
                
                print("\n Hypothesis Testing Checks ")
                print("1. T-test (P>|t|): Is this specific feature actually useful? (Want < 0.05)")
                print("2. F-test (Prob F-statistic): Is the whole model garbage? (Want < 0.05)")
                
                plt.figure(figsize=(14, 6))
                # Plot a snippet of training data
                train_plot_n = 300
                y_train_subset = y_train.iloc[-train_plot_n:] if len(y_train) > train_plot_n else y_train
                x_train = np.arange(len(y_train_subset))
                plt.scatter(x_train, y_train_subset, color='gray', alpha=0.5, s=15, label='Train Data (Sample)')
                
                # Plot the Test predictions vs Actual
                x_test = np.arange(len(y_train_subset), len(y_train_subset) + len(y_test))
                plt.plot(x_test, y_test, color='blue', label='Test Data (Actual)', linewidth=1.5)
                plt.plot(x_test, y_pred, color='red', linestyle='--', label='Predicted', linewidth=1.5)
                
                plt.title(f"Example Regression: {target_col.capitalize()} (Subject {subj})")
                plt.xlabel("Samples (Time)")
                plt.ylabel(target_col.capitalize())
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

        #  2. Print the Scorecard 
        results_df = pd.DataFrame(results_list)
        print(f"\n================ FINAL VALIDATION METRICS: {target_col.upper()} ================")
        print(results_df.round(4).to_string(index=False))
        
        #  3. Average everything to get a single performance number 
        print("\n Overall Model Performance (Average across all folds) ")
        avg_metrics = results_df.mean(numeric_only=True)
        print(avg_metrics)

        #  4. Scatter Plot: Truth vs Prediction 
        plt.figure(figsize=(8, 8))
        plt.scatter(curr_target_true, curr_target_pred, alpha=0.3, color='purple', s=10)
        
        min_val = min(min(curr_target_true), min(curr_target_pred))
        max_val = max(max(curr_target_true), max(curr_target_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        
        plt.title(f"Overall Validation: Predicted vs Actual {target_col.capitalize()}")
        plt.xlabel(f"Actual {target_col.capitalize()}")
        plt.ylabel(f"Predicted {target_col.capitalize()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # 
    #  5. THE EMOTION MAP (Russell's Circumplex) 
    print(f"\n{'='*60}")
    print("VISUALIZATION: RUSSELL'S CIRCUMPLEX MODEL (VALENCE vs AROUSAL)")
    print(f"{'='*60}")
    
    val_true = np.array(circumplex_data['valence']['true'])
    val_pred = np.array(circumplex_data['valence']['pred'])
    aro_true = np.array(circumplex_data['arousal']['true'])
    aro_pred = np.array(circumplex_data['arousal']['pred'])
    
    plt.figure(figsize=(10, 10))
    
    # Drawing the Crosshairs for the Quadrants
    # Assuming standard SAM scale (1-9), the center is 5.
    center_x = 5 if np.mean(val_true) > 2 else 0
    center_y = 5 if np.mean(aro_true) > 2 else 0
    
    plt.axhline(y=center_y, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=center_x, color='k', linestyle='-', alpha=0.5)
    
    # Labeling the 4 Emotional Zones
    plt.text(center_x + 1, center_y + 1, "High Arousal / Positive (Excited)", fontsize=10, color='gray')
    plt.text(center_x - 1, center_y + 1, "High Arousal / Negative (Stressed)", fontsize=10, color='gray', ha='right')
    plt.text(center_x - 1, center_y - 1, "Low Arousal / Negative (Depressed)", fontsize=10, color='gray', ha='right')
    plt.text(center_x + 1, center_y - 1, "Low Arousal / Positive (Relaxed)", fontsize=10, color='gray')

    # Plot a random subset so the graph isn't a solid blob of ink
    subset_idx = np.random.choice(len(val_true), size=min(2000, len(val_true)), replace=False)
    
    plt.scatter(val_true[subset_idx], aro_true[subset_idx], c='blue', alpha=0.3, label='Ground Truth', s=20)
    plt.scatter(val_pred[subset_idx], aro_pred[subset_idx], c='red', alpha=0.3, label='Predicted', marker='x', s=20)
    
    plt.title("Russell's Circumplex Model: Emotion Recognition Performance")
    plt.xlabel("Valence (Unpleasant <--> Pleasant)")
    plt.ylabel("Arousal (Deactivated <--> Activated)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    X_final, y_reg, y_class, groups = run_pipeline()
    
    run_phase_2_regression(X_final, y_reg, groups)