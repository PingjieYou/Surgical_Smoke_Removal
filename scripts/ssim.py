import numpy as np

# Given parameters
mean = 90.600
variance = 6.76
std_dev = np.sqrt(variance)

# Generate 10 random numbers from normal distribution
random_numbers = np.random.normal(loc=mean, scale=std_dev, size=10)

# Adjust to ensure the final mean is exactly 78
adjusted_numbers = random_numbers - (np.mean(random_numbers) - mean)

# Verify the final mean
final_mean = np.mean(adjusted_numbers)
print(adjusted_numbers, final_mean)
