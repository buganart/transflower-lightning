import numpy as np
import pickle

# refactor later for cmd "filename" and "scaler" input

#filename
filename = "Medium_dance_classical.feats_ddcpca.npy"

#sklearn StandardScaler (n_features=85)
scale_filename = 'violin_scalar_3files.pkl'
scaler = pickle.load(open(scale_filename, "rb"))

# shape: (n, 85)
combined_feat = np.load(filename)
scaled_feat = (combined_feat - scaler.mean_) / scaler.scale_

#split into 2 files
filename_base = filename.split(".")[0]
    #mel80
np.save(f"{filename_base}.audio_mel80.npy", scaled_feat[:,:80])
    #beats5
np.save(f"{filename_base}.audio_beats5.npy", scaled_feat[:,80:])