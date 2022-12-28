import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Shape of the color-coded adjacency tensor
N = 100
M = 15
K = 15

# Create some random data (Replace with actual data when required)
volume = np.random.randint(1, 5, size=(N, M, K))

# Broadcast arrays
x = np.arange(volume.shape[0])[:, None, None]
y = np.arange(volume.shape[1])[None, :, None]
z = np.arange(volume.shape[2])[None, None, :]
x, y, z = np.broadcast_arrays(x, y, z)

# Turn the volumetric data into an RGB array that's
# just grayscale.
c = volume.ravel()

# Plot the tensor
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x.ravel(),
           y.ravel(),
           z.ravel(),
           c=c)
plt.axis('off')
plt.show()
None