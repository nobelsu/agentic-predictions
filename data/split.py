import pandas as pd
from sklearn.model_selection import train_test_split

# --- SETTINGS ---
input_csv = "data.csv"       # your original CSV
train_csv = "train.csv"      # output train CSV
test_csv = "test.csv"        # output test CSV
test_size = 0.08              # 8% for testing
random_state = 67            # for reproducibility
shuffle = True               # shuffle before splitting

# --- LOAD DATA ---
df = pd.read_csv(input_csv)

# --- SPLIT DATA ---
train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    shuffle=shuffle
)

# --- SAVE OUTPUT ---
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Train set saved to {train_csv}, shape: {train_df.shape}")
print(f"Test set saved to {test_csv}, shape: {test_df.shape}")
