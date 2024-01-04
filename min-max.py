from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Updated sample data
data = np.array([[0.5, 1.5, 2.5],
                 [3.5, 4.5, 5.5],
                 [6.5, 7.5, 8.5]])

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("\n Symbol no.: 24001")
print("Original Data:\n", data)
print("\nScaled Data (Min-Max scaled between [0, 1]):\n", scaled_data)
 