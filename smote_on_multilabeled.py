import os
import numpy as np
from imblearn.over_sampling import SMOTE

# Directory containing the .npy files
dir_path = "/home/group10/deephalo_gnn/Labeled subhalo matrices of haloes/train"
dir_path2 = "/home/group10/deephalo_gnn/Imbalance_Resampled_for_mulltilabel/train"

# Create a SMOTE object
smote = SMOTE()

# Iterate over each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".npy") and int(filename[:-4])>50:
        # Load the data from the .npy file
        data = np.load(os.path.join(dir_path, filename))

        # Separate the features and the labels (assuming labels are in the last column)
        X = data[:, :-1]
        y = data[:, -1]
        
        if len(np.unique(y)) == 1:
            print("skipping:",filename)
            continue

        # Apply SMOTE on the data
        X_res, y_res = smote.fit_resample(X, y)

        # Concatenate the resampled features and labels
        data_res = np.column_stack((X_res, y_res))

        # Save the resampled data back to the .npy file
        np.save(os.path.join(dir_path2, filename), data_res)
        print("Resampled:",filename)