import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# refactor later for cmd "filename" and "scaler" input

# filename
data_path = "./"

filename_list = Path(data_path).rglob("*.feats_ddcpca.npy")

# #sklearn StandardScaler (n_features=85)
# scale_filename = 'violin_scalar_3files.pkl'
# scaler = pickle.load(open(scale_filename, "rb"))

scaler = StandardScaler()
filename_array = []
data = []
for filename in filename_list:
    filename_base = str(filename.stem).split(".")[0]  # str(filename.parent)
    # shape: (n, 85)
    combined_feat = np.load(filename)
    filename_array.append((filename_base, combined_feat.shape[0]))
    data.append(combined_feat)

data = np.concatenate(data, axis=0)
scaler.fit(data)
pickle.dump(scaler, open("feat_scalar.pkl", "wb"))

data = scaler.transform(data)
# data = (data - scaler.mean_) / scaler.scale_

for filename_base, length in filename_array:
    print(filename_base)
    scaled_feat = data[:length]
    data = data[length:]
    # mel80
    np.save(f"{filename_base}.audio_mel80.npy", scaled_feat[:, :80])
    # beats5
    np.save(f"{filename_base}.audio_beats5.npy", scaled_feat[:, 80:])
