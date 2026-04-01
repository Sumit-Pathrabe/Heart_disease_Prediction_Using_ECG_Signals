import wfdb
import matplotlib.pyplot as plt
import os

# Update this to point to one of your unzipped .mat/.hea pairs
# Don't include the extension, just the record name (e.g., 'JS00001')
sample_record = 'path/to/your/data/JS00001' 

# 1. Read the record
record = wfdb.rdrecord(sample_record)
print(f"Patient Metadata: {record.comments}")
print(f"Sampling Frequency: {record.fs} Hz")

# 2. Plot the first 1000 samples (2 seconds at 500Hz)
# We'll plot Lead II (usually index 1)
plt.figure(figsize=(12, 4))
plt.plot(record.p_signal[:1000, 1], color='red', linewidth=1)
plt.title(f"ECG Lead II - Record: {os.path.basename(sample_record)}")
plt.xlabel("Samples")
plt.ylabel("Voltage (mV)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()