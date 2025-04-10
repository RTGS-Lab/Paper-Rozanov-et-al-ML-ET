import numpy as np
def create_rolling_windows(df, window_size=30):
    df = df.sort_values(["site", "TIMESTAMP"])
    
    feature_cols = ['G_F_MDS', 'TA_F', 'VPD_F', 'WS_F', 'PA_F', 'H_F_MDS']
    
    # Function to apply rolling window per site
    def site_rolling(site_df):
        X = np.lib.stride_tricks.sliding_window_view(site_df[feature_cols].values, (window_size, len(feature_cols)))
        return X[:, 0, :, :]  # Extract the correct axis

    # Apply rolling per site and concatenate results
    windows = np.concatenate([site_rolling(group) for _, group in df.groupby("site")], axis=0)

    return windows