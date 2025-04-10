import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 **Step 1: Load .npy Rainfall Data**
file_path = "/path/to/train_rain_224.npy"  # ⚡ Update this path
rain_data = np.load(file_path)  # Load NumPy array

# 🔹 **Step 2: Compute M/N Ratio**
M = np.sum(rain_data > 0)  # Count pixels with rain
N = rain_data.size  # Total pixels
imbalance_ratio = (M / N) * 100

# 🔹 **Step 3: Print Results**
print(f"✅ Rainfall Pixels (M) = {M}")
print(f"✅ Total Pixels (N) = {N}")
print(f"✅ Imbalance Ratio = {imbalance_ratio:.2f}%")

# 🔹 **Step 4: Plot Histogram**
plt.figure(figsize=(8,5))
sns.histplot(rain_data.flatten(), bins=50, kde=True, log_scale=(False, True), color="green", edgecolor="black")
plt.xlabel("Rainfall (mm/h)")
plt.ylabel("Frequency (Log Scale)")
plt.title(f"Rainfall Data Distribution (Imbalance = {imbalance_ratio:.2f}%)")
plt.grid()

# 🔹 **Step 5: Save Plot**
save_path = "/path/to/save/imbalance_plot.png"  # ⚡ Update path
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Image saved at: {save_path}")

plt.show()

