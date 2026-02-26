import pandas as pd
import numpy as np
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
import warnings
import os
import pickle

BASE_DIR = "WESAD/" 

# ignore warnings so the terminal doesn't get flooded with spam during the loop
warnings.filterwarnings('ignore')

def convert_acc(chest_dict, wrist_dict):
    """
    Fixing the units so both sensors are actually comparable (m/s^2).
    """
    standard_gravity = 9.80665

    # Chest data is already in 'g' units, so just multiply by gravity to get standard metric
    chest_dict['ACC'] = chest_dict['ACC'] * standard_gravity

    # Wrist data is messier (raw 8-bit ints). 
    # The E4 manual says to divide by 64 and multiply by gravity to normalize it.
    wrist_dict['ACC'] = (wrist_dict['ACC'] / 64.0) * standard_gravity

    return chest_dict, wrist_dict

def parse_wesad_quest(subject_id, base_dir):
    """
    Reads those messy questionnaire CSVs to figure out exactly when 
    the stress/baseline tasks started and ended.
    """
    file_path = os.path.join(base_dir, subject_id, f"{subject_id}_quest.csv")
    
    # Try different separators because some CSVs are formatted weirdly
    try:
        df_raw = pd.read_csv(file_path, sep=';', header=None)
    except:
        df_raw = pd.read_csv(file_path, sep=',', header=None)

    def get_row_idx(label):
        return df_raw.index[df_raw[0] == label].tolist()

    # Find the order of tasks (Baseline, Stress, Amusement, etc.)
    order_row_idx = get_row_idx('# ORDER')[0]
    order_values = df_raw.iloc[order_row_idx, 1:].values
    
    valid_conditions = {}
    ignored_labels = ['nan'] 
    
    for i, val in enumerate(order_values):
        val_str = str(val).strip()
        if val_str.lower() != 'nan':
            valid_conditions[i + 1] = val_str
            
    # Helper to convert the minute.seconds format into total seconds
    def parse_time(val):
        try:
            val = float(val)
            minutes = int(val)
            seconds = round((val - minutes) * 100)
            return minutes * 60 + seconds
        except:
            return np.nan

    start_row = df_raw.iloc[get_row_idx('# START')[0], :]
    end_row = df_raw.iloc[get_row_idx('# END')[0], :]
    
    # Map the start/end times to the specific condition names
    timing_data = []
    for col_idx, cond_name in valid_conditions.items():
        timing_data.append({
            'Condition': cond_name,
            'Start_Sec': parse_time(start_row[col_idx]),
            'End_Sec': parse_time(end_row[col_idx])
        })
    
    df_conditions = pd.DataFrame(timing_data).set_index('Condition')

    # Grab the self-reported stress/emotion scores (SAM scores)
    sam_rows = get_row_idx('# DIM')
    sam_data = []
    
    if sam_rows:
        # Ignore reading tasks since they don't have emotion scores associated with them
        conds_with_scores = [c for c in valid_conditions.values() if c not in ['sRead', 'fRead']]
        
        start_r = sam_rows[0]
        for i, cond in enumerate(conds_with_scores):
            # sanity check to make sure we don't go out of bounds
            if (start_r + i) < len(df_raw):
                val = float(df_raw.iloc[start_r + i, 1])
                aro = float(df_raw.iloc[start_r + i, 2])
                sam_data.append({'Condition': cond, 'Valence': val, 'Arousal': aro})
            
    df_sam = pd.DataFrame(sam_data).set_index('Condition')

    # combine timing info with the emotion scores
    subject_meta = df_conditions.join(df_sam, how='left')
    
    # Create a binary target: 1 if it's the stress task (TSST), 0 for everything else
    subject_meta['Label'] = subject_meta.index.map(lambda x: 1 if x == 'TSST' else 0)
    
    subject_meta['subject'] = subject_id
    
    return subject_meta 

# Just some standard signal processing functions needed for feature extraction

def get_peak_freq(sig, fs):
    f, Pxx = welch(sig, fs, nperseg=min(len(sig), 256))
    return f[np.argmax(Pxx)]

def get_absolute_integral(sig):
    return np.sum(np.abs(sig))

def get_spectral_energy(sig, fs, bands):
    f, Pxx = welch(sig, fs, nperseg=min(len(sig), 256))
    energy = {}
    for name, (low, high) in bands.items():
        idx = np.logical_and(f >= low, f <= high)
        energy[name] = np.trapz(Pxx[idx], f[idx])
    return energy

def get_slope(sig):
    return np.polyfit(np.arange(len(sig)), sig, 1)[0] if len(sig) > 1 else 0

def get_dynamic_range(sig):
    return np.max(sig) - np.min(sig)

def get_hrv_features(peaks, fs):
    if len(peaks) < 2: return {}
    rr_intervals = np.diff(peaks) / fs * 1000 # convert to ms
    return {
        'mean_RR': np.mean(rr_intervals),
        'std_RR': np.std(rr_intervals),
        'RMSSD': np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    }

# MAIN FEATURE EXTRACTION 
def extract_wesad_features(df_chest, wrist_dict):
    """
    The heavy lifting. Sliding window over the raw signals to calculate stats (mean, std, etc).
    Syncs the 700Hz chest data with the slower wrist data.
    """
    
    # Sampling rates defined in the readme
    FS_CHEST = 700
    FS_WRIST = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
    
    # Using window sizes from the original paper
    SHIFT_SEC = 0.25
    WIN_ACC_SEC = 5.0   # Short window for movement
    WIN_BIO_SEC = 60.0  # Long window for heart/skin/temp
    
    step_chest = int(SHIFT_SEC * FS_CHEST)
    
    final_rows = []
    
    total_len = len(df_chest)
    
    # Skip the first minute so we have enough data for the first full window
    start_idx = int(WIN_BIO_SEC * FS_CHEST)
    
    print(f"Starting feature extraction. Total steps: {(total_len - start_idx) // step_chest}")

    for i in range(start_idx, total_len, step_chest):
        
        # We need a master timestamp for this window to sync everything later
        if 'time' in df_chest.columns:
            current_ts = df_chest['time'].iloc[i]
        else:
            current_ts = i / FS_CHEST
            
        row = {'time': current_ts}
        
            
        # Get the last 5 seconds of ACC data
        win_acc = int(WIN_ACC_SEC * FS_CHEST)
        slc_acc = slice(i - win_acc, i)
        
        for axis in ['x', 'y', 'z']:
            col_name = f'ACC_{axis}'
            if col_name in df_chest:
                sig = df_chest[col_name].iloc[slc_acc].values
                row[f'Chest_ACC_{axis}_mean'] = np.mean(sig)
                row[f'Chest_ACC_{axis}_std'] = np.std(sig)
                row[f'Chest_ACC_{axis}_peakFreq'] = get_peak_freq(sig, FS_CHEST)
                row[f'Chest_ACC_{axis}_integral'] = get_absolute_integral(sig)
        
        # Combine X, Y, Z into magnitude just in case orientation shifts
        if all(c in df_chest for c in ['ACC_x', 'ACC_y', 'ACC_z']):
            acc_x = df_chest['ACC_x'].iloc[slc_acc].values
            acc_y = df_chest['ACC_y'].iloc[slc_acc].values
            acc_z = df_chest['ACC_z'].iloc[slc_acc].values
            acc_3d = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            row['Chest_ACC_3D_mean'] = np.mean(acc_3d)
            row['Chest_ACC_3D_integral'] = get_absolute_integral(acc_3d)

        # EMG (Muscle activity) - also uses 5 sec window
        if 'EMG' in df_chest:
            sig = df_chest['EMG'].iloc[slc_acc].values
            row['Chest_EMG_mean'] = np.mean(sig)
            row['Chest_EMG_std'] = np.std(sig)
            row['Chest_EMG_integral'] = get_absolute_integral(sig)
            row['Chest_EMG_peakFreq'] = get_peak_freq(sig, FS_CHEST)
            
            bands = {'EMG_0_50': (0,50), 'EMG_50_100': (50,100)}
            ens = get_spectral_energy(sig, FS_CHEST, bands)
            for k, v in ens.items(): row[f'Chest_{k}'] = v

        # PHYSIO SIGNALS (60 sec window)
        win_bio = int(WIN_BIO_SEC * FS_CHEST)
        slc_bio = slice(i - win_bio, i)
        
        # Temperature
        if 'Temp' in df_chest: 
            sig = df_chest['Temp'].iloc[slc_bio].values
            row['Chest_Temp_mean'] = np.mean(sig)
            row['Chest_Temp_std'] = np.std(sig)
            row['Chest_Temp_min'] = np.min(sig)
            row['Chest_Temp_max'] = np.max(sig)
            row['Chest_Temp_slope'] = get_slope(sig)
            row['Chest_Temp_range'] = get_dynamic_range(sig)

        # EDA (Skin Conductance)
        if 'EDA' in df_chest:
            sig = df_chest['EDA'].iloc[slc_bio].values
            row['Chest_EDA_mean'] = np.mean(sig)
            row['Chest_EDA_std'] = np.std(sig)
            row['Chest_EDA_min'] = np.min(sig)
            row['Chest_EDA_max'] = np.max(sig)
            row['Chest_EDA_slope'] = get_slope(sig)
            
            # Try to separate phasic/tonic components with a filter
            try:
                b, a = butter(2, 0.05 / (FS_CHEST/2), btype='low')
                scl = filtfilt(b, a, sig)
                scr = sig - scl
                row['Chest_SCL_mean'] = np.mean(scl)
                row['Chest_SCR_std'] = np.std(scr)
            except:
                pass # if the signal is garbage/too short, just skip

        # Respiration
        if 'Resp' in df_chest:
            sig = df_chest['Resp'].iloc[slc_bio].values
            centered = sig - np.mean(sig)
            peaks, _ = find_peaks(centered, distance=FS_CHEST*2)
            row['Chest_Resp_rate'] = len(peaks)
            
        # ECG (Heart Rate stuff)
        if 'ECG' in df_chest:
            sig = df_chest['ECG'].iloc[slc_bio].values
            # Peak detection tuned for ECG R-peaks
            peaks, _ = find_peaks(sig, height=np.mean(sig)*1.5, distance=FS_CHEST*0.4)
            hrv_metrics = get_hrv_features(peaks, FS_CHEST)
            for k, v in hrv_metrics.items(): row[f'Chest_ECG_{k}'] = v

        # WRIST SENSORS (Syncing different sampling rates) 
        curr_time_sec = current_ts
        
        # 1. Wrist ACC (32 Hz)
        idx_w_acc = int(curr_time_sec * FS_WRIST['ACC'])
        win_w_acc = int(WIN_ACC_SEC * FS_WRIST['ACC'])
        
        if idx_w_acc < len(wrist_dict['ACC']):
            start = max(0, idx_w_acc - win_w_acc)
            seg = wrist_dict['ACC'][start : idx_w_acc]
            
            # Standard WESAD format is (N, 3)
            if seg.ndim == 2 and seg.shape[1] == 3:
                for ax_idx, ax_name in enumerate(['x', 'y', 'z']):
                    vals = seg[:, ax_idx]
                    row[f'Wrist_ACC_{ax_name}_mean'] = np.mean(vals)
                    row[f'Wrist_ACC_{ax_name}_std'] = np.std(vals)
                    row[f'Wrist_ACC_{ax_name}_peakFreq'] = get_peak_freq(vals, FS_WRIST['ACC'])
                
                acc_3d_w = np.sqrt(seg[:,0]**2 + seg[:,1]**2 + seg[:,2]**2)
                row['Wrist_ACC_3D_mean'] = np.mean(acc_3d_w)

        # 2. Wrist TEMP (4 Hz)
        idx_w_tmp = int(curr_time_sec * FS_WRIST['TEMP'])
        win_w_tmp = int(WIN_BIO_SEC * FS_WRIST['TEMP'])
        if idx_w_tmp < len(wrist_dict['TEMP']):
            start = max(0, idx_w_tmp - win_w_tmp)
            seg = wrist_dict['TEMP'][start : idx_w_tmp].flatten()
            if len(seg) > 0:
                row['Wrist_Temp_mean'] = np.mean(seg)
                row['Wrist_Temp_std'] = np.std(seg)
                row['Wrist_Temp_range'] = get_dynamic_range(seg)

        # 3. Wrist EDA (4 Hz)
        idx_w_eda = int(curr_time_sec * FS_WRIST['EDA'])
        win_w_eda = int(WIN_BIO_SEC * FS_WRIST['EDA'])
        if idx_w_eda < len(wrist_dict['EDA']):
            start = max(0, idx_w_eda - win_w_eda)
            seg = wrist_dict['EDA'][start : idx_w_eda].flatten()
            if len(seg) > 0:
                row['Wrist_EDA_mean'] = np.mean(seg)
                row['Wrist_EDA_slope'] = get_slope(seg)

        # 4. Wrist BVP (64 Hz)
        idx_w_bvp = int(curr_time_sec * FS_WRIST['BVP'])
        win_w_bvp = int(WIN_BIO_SEC * FS_WRIST['BVP'])
        if idx_w_bvp < len(wrist_dict['BVP']):
            start = max(0, idx_w_bvp - win_w_bvp)
            seg = wrist_dict['BVP'][start : idx_w_bvp].flatten()
            if len(seg) > 0:
                row['Wrist_BVP_mean'] = np.mean(seg)
                peaks, _ = find_peaks(seg, distance=FS_WRIST['BVP']*0.4)
                row['Wrist_BVP_rate'] = len(peaks)

        final_rows.append(row)

    return pd.DataFrame(final_rows)

def add_labels_to_features(features_df, subject_meta, time_col='time'):
    """
    Takes the features we just calculated and attaches the Label/Valence/Arousal 
    based on the timestamp. Drops any data that doesn't fall into a valid task.
    """
    features_df['label'] = np.nan
    features_df['valence'] = np.nan
    features_df['arousal'] = np.nan
    
    if not subject_meta.empty:
        features_df['subject'] = subject_meta['subject'].iloc[0]

        
    # Loop through the tasks (Base, TSST, Fun, etc.)
    for condition, row in subject_meta.iterrows():
        start = row['Start_Sec']
        end = row['End_Sec']
        
        # If there's no emotion score (like the reading tasks), skip it
        if pd.isna(row['Valence']):
            continue
            
        # grab all rows that happened during this task
        mask = (features_df[time_col] >= start) & (features_df[time_col] <= end)
        
        features_df.loc[mask, 'label'] = int(row['Label'])
        features_df.loc[mask, 'valence'] = row['Valence']
        features_df.loc[mask, 'arousal'] = row['Arousal']
        
    # Get rid of the transition periods where we don't have labels
    final_df_labeled = features_df.dropna(subset=['label', 'valence', 'arousal']).copy()
    
    final_df_labeled['label'] = final_df_labeled['label'].astype(int)
    
    return final_df_labeled


def create_chest_df(chest_dict):
    """
    Flattens the nested dictionary into a simple Pandas DataFrame.
    """
    # Start with ECG since it's the main signal
    df = pd.DataFrame(data=chest_dict['ECG'].flatten(), columns=['ECG'])
    
    # All these share the same 700Hz clock, so we can just column bind them
    df['EMG'] = chest_dict['EMG'].flatten()
    df['EDA'] = chest_dict['EDA'].flatten()
    df['Temp'] = chest_dict['Temp'].flatten()
    df['Resp'] = chest_dict['Resp'].flatten()
    
    # ACC comes in 3 columns (x, y, z), so split them out
    acc_data = chest_dict['ACC']
    df['ACC_x'] = acc_data[:, 0]
    df['ACC_y'] = acc_data[:, 1]
    df['ACC_z'] = acc_data[:, 2]
    
    # Create a time column so we can sync later (0 to N seconds)
    fs = 700 # Hz
    df['time'] = df.index / fs 
    
    return df


if __name__ == "__main__":
    print("Starting WESAD Data Processing Pipeline...")
    all_subjects_data = []

    # Skip S1 and S12 (bad data)
    subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 
                'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

    for subj in subjects:
        print(f"Processing {subj}...")
        

        subject_path = os.path.join(BASE_DIR, subj)
        pkl_path = os.path.join(subject_path, f"{subj}.pkl")

        
        with open(pkl_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        
        chest_dict = data['signal']['chest']
        wrist_dict = data['signal']['wrist']

        # Fix units
        chest_dict, wrist_dict = convert_acc(chest_dict, wrist_dict)
        
        # Convert dict to dataframe
        df_chest = create_chest_df(chest_dict)
        
        print("  Extracting Features...")
        # Run sliding window
        df_features = extract_wesad_features(df_chest, wrist_dict)
        
        print("  Parsing Metadata...")
        # Get start/end times for tasks
        subject_meta = parse_wesad_quest(subj, BASE_DIR)
        
        print("  Merging Labels...")
        # Sync features with labels
        df_labeled = add_labels_to_features(df_features, subject_meta, time_col='time')
        
        all_subjects_data.append(df_labeled)

    # Smash everyone into one giant DataFrame
    master_df = pd.concat(all_subjects_data, ignore_index=True)

    time_diff = master_df['time'].diff().fillna(0)

    # Kind of a hack: Since we just concatenated everyone, we look for 
    # where the time variable resets (drops by > 100s) to identify the next subject.
    split_indices = master_df.index[time_diff < -100].tolist()

    # Define the slice points for each subject
    slice_points = [0] + split_indices + [len(master_df)]

    # Sanity check to make sure we didn't mess up the alignment
    detected_count = len(slice_points) - 1
    expected_count = len(subjects)

    print(f"Detected {detected_count} subject blocks.")
    print(f"Expected {expected_count} subjects.")

    if detected_count == expected_count:
        print("Alignment perfect! Assigning IDs...")
        
        master_df['subject'] = None
        
        # Loop through our slice points and assign the correct subject ID
        for i in range(len(subjects)):
            subj_name = subjects[i]
            start_idx = slice_points[i]
            end_idx = slice_points[i+1]
            
            master_df.iloc[start_idx:end_idx, master_df.columns.get_loc('subject')] = subj_name
            
        print("Done! 'subject' column added.")
        print(master_df[['time', 'subject']].head())
        # Show the crossover point to prove it worked
        print(master_df[['time', 'subject']].iloc[split_indices[0]-5 : split_indices[0]+5]) 
    else:
        print("CRITICAL WARNING: The number of detected time-resets does not match your subject list.")
        print("Did you process a subject that isn't in the list, or is the list order wrong?")
        print(f"Split indices found at: {split_indices}")


    master_df.to_csv("wesad_features.csv", index=False)


    print("Final shape:", master_df.shape)
