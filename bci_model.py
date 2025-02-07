import numpy as np
import mne
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Step 1: Load EEG Data ---
# Ensure you have an EEG dataset file at 'data/eeg_data.fif'
try:
    data = mne.io.read_raw_fif('data/eeg_data.fif', preload=True)
except Exception as e:
    print("Error loading EEG data:", e)
    exit()

# Apply bandpass filter between 1-50 Hz.
data.filter(1, 50)

# --- Step 2: Preprocess & Feature Extraction ---
def extract_features(eeg_data):
    # Use FFT to extract frequency domain features.
    fft_result = np.fft.fft(eeg_data)
    return np.abs(fft_result)

# Get raw EEG data.
eeg_data = data.get_data()
features = extract_features(eeg_data)

# Reshape features for the CNN.
# For demonstration, assuming features is 2D; we add a channel dimension.
features = features.reshape(features.shape[0], features.shape[1], 1)

# Normalize features.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)

# Dummy labels (replace with actual labels from your dataset).
y_train = np.random.randint(0, 2, features_scaled.shape[0])

# --- Step 3: Build the CNN Model ---
input_shape = (features_scaled.shape[1], features_scaled.shape[2], 1)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Binary classification
])

# Compile the model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Step 4: Train the Model ---
print("Training the model...")
model.fit(features_scaled, y_train, epochs=10, batch_size=32)

# --- Step 5: Evaluate & Visualize Results ---
print("Evaluating the model...")
loss, accuracy = model.evaluate(features_scaled, y_train, verbose=0)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Plot a sample EEG signal from the first channel.
plt.figure(figsize=(10, 4))
plt.plot(eeg_data[0])
plt.title("Sample EEG Signal - Channel 1")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Save the trained model.
model.save("results/bci_model.h5")
print("Model saved to 'results/bci_model.h5'")
