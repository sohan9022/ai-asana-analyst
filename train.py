import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def train_model():
    print("🚀 Starting MLOps Automated Training Pipeline...")
    
    # 1. Define Paths to your specific Train/Test folders
    train_dir = os.path.join("dataset", "train")
    test_dir = os.path.join("dataset", "test")
    
    img_height = 75
    img_width = 75
    batch_size = 32

    # 2. Load the Training and Validation Data explicitly
    print("Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    print("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Normalize pixel values to be between 0 and 1
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 3. Define the CNN Architecture 
    print("Building model architecture...")
    model = Sequential([
        Input(shape=(75, 75, 3)), # Fixed the Keras Input shape warning
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax') # Restored to 5 classes for your 5 poses!
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train the Model
    print("Training model...")
    epochs = 15 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 5. Save the Model to your App Engine
    save_dir = "app/engine"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "yoga_pose_model.keras")
    
    model.save(model_path)
    print(f"✅ Training complete! New model automatically saved to {model_path}")

if __name__ == "__main__":
    train_model()

# Testing the MLOps pipeline automation!
print("Training complete. Passing to evaluation...")