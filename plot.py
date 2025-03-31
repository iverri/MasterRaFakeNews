import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters from your model
mean = 0.5
std_dev = 0.15

# Generate a large sample of values
num_samples = 100
raw_values = [random.gauss(mean, std_dev) for _ in range(num_samples)]
truncated_values = [min(max(val, 0), 1) for val in raw_values]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot 1: Original Gaussian distribution
plt.subplot(1, 2, 1)
plt.hist(raw_values, bins=50, alpha=0.7, color='blue')
plt.title('Original Gaussian Distribution\nμ=0.5, σ=0.15')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.axvline(x=mean, color='red', linestyle='--', label='Mean (0.5)')
plt.axvline(x=mean-std_dev, color='green', linestyle='--', label='Mean ± 1σ')
plt.axvline(x=mean+std_dev, color='green', linestyle='--')
plt.legend()

# Plot 2: Truncated distribution (as used in the model)
plt.subplot(1, 2, 2)
plt.hist(truncated_values, bins=50, alpha=0.7, color='purple')
plt.title('Truncated Gaussian Distribution\n(Values clamped between 0 and 1)')
plt.xlabel('Credibility Level')
plt.ylabel('Frequency')
plt.axvline(x=mean, color='red', linestyle='--', label='Mean (0.5)')
plt.axvline(x=mean-std_dev, color='green', linestyle='--', label='Mean ± 1σ')
plt.axvline(x=mean+std_dev, color='green', linestyle='--')
plt.legend()

plt.tight_layout()
plt.savefig('credibility_distribution.png')
plt.show()

# Print some statistics
print(f"Original distribution:")
print(f"  Mean: {np.mean(raw_values):.4f}")
print(f"  Std Dev: {np.std(raw_values):.4f}")
print(f"  Min: {min(raw_values):.4f}")
print(f"  Max: {max(raw_values):.4f}")
print(f"\nTruncated distribution (as used in model):")
print(f"  Mean: {np.mean(truncated_values):.4f}")
print(f"  Std Dev: {np.std(truncated_values):.4f}")
print(f"  Min: {min(truncated_values):.4f}")
print(f"  Max: {max(truncated_values):.4f}")
print(f"  % at min (0): {truncated_values.count(0) / num_samples * 100:.2f}%")
print(f"  % at max (1): {truncated_values.count(1) / num_samples * 100:.2f}%")