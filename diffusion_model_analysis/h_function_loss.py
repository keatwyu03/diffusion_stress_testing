epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]
losses = [11243.747138, 19529.561693, 19855.813083, 15580.585849, 10961.704964, 12462.725433, 12654.617460, 11217.303379, 13786.617295, 14961.165851, 26745.683765, 11402.252886, 15625.691264, 15250.557010, 26681.974539, 19224.694123, 13477.372408, 10522.173946, 10086.328421, 19478.883885, 15209.145077, 21921.905170, 14539.233737, 12521.928990]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Score Function Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/diffusion_training_loss.png", dpi=150, bbox_inches="tight")
plt.show()