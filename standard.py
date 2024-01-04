from sklearn.preprocessing import StandardScaler
import numpy as np

# Updated sample data
data = np.array([[0.5, 1.5, 2.5],
                 [3.5, 4.5, 5.5],
                 [6.5, 7.5, 8.5]])

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("\nSymbol no.: 24001")

print("Original Data:\n", data)
print("\nScaled Data:\n", scaled_data)
