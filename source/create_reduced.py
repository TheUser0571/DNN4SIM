import numpy as np
import sys

# Check usage
if len(sys.argv) != 3:
    print("Usage:\n\t>>python create_reduced in_path out_path\n")
    exit()      
# Process input parameters
feat_path = sys.argv[1]
out_path = sys.argv[2]

print('Loading data...', end='\r')
# Load data
features = []
with open(feat_path, 'rb') as f:
    features = np.load(f)
print('Loading data... Done.')
features = features[features.shape[0]//2:]
print('Saving reduced data...', end='\r')
with open(out_path, 'wb') as f:
    np.save(f, features)
    
print('Saving reduced... Done.')