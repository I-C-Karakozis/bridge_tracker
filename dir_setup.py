import os

### ---- INITIALIZATION ---- ###

BASE_DIR = "baseline_test_data"

### ------------------------ ###

# create top directories
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# create dir - labels for baseline test-data
suit = ["H","D","S","C"]
value = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
for v in value:
    for s in suit:
        new_dir = os.path.join(BASE_DIR, v+s)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
