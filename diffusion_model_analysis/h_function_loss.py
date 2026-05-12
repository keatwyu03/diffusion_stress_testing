epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
losses = [189.244450, 180.639615, 176.222994, 174.599042, 173.470610, 172.434112, 171.484129, 171.158564, 170.655362, 170.456893, 170.254056, 168.824231, 168.709563, 167.778071, 167.706216]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Score Function Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()