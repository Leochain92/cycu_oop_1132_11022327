import numpy as np

# Read the 2nd_floor_displacement.txt file (specify full path)
filename = r"C:\Users\truck\Downloads\2nd_floor_displacement.txt" # Ensure this path is correct
data = []

with open(filename, "r", encoding="utf-8") as file:
    for line in file:
        try:
            # Assuming each line has time and displacement, converting them to float
            # Note: The variable name 'acc' is used here, but it represents 'displacement' from your file.
            time, displacement = map(float, line.split())
            data.append(displacement) # Append displacement data
        except ValueError:
            # Ignore lines with format errors (e.g., empty lines or non-numeric lines)
            continue

# Convert to a NumPy array
# The variable name 'acceleration' is used for consistency with your previous code,
# but it now holds displacement data.
displacement_array = np.array(data)

# Calculate relevant displacement metrics
mean_displacement = np.mean(displacement_array)  # Mean value (can be close to 0)
abs_mean_displacement = np.mean(np.abs(displacement_array)) # Average Absolute Displacement
rms_displacement = np.sqrt(np.mean(displacement_array**2))  # Root Mean Square (RMS) Displacement
peak_displacement = np.max(np.abs(displacement_array))  # Peak Displacement

# Display results
# Note: The unit 'g' (for acceleration) is not appropriate for displacement.
# You'll need to know the unit of your displacement data (e.g., mm, cm, inches).
# For now, I'll use 'units' as a placeholder.
print(f"平均位移 (Mean Displacement): {mean_displacement:.6f} units")
print(f"平均絕對位移 (Average Absolute Displacement): {abs_mean_displacement:.6f} units")
print(f"均方根值位移 (RMS Displacement): {rms_displacement:.6f} units")
print(f"尖峰位移 (Peak Displacement): {peak_displacement:.6f} units")