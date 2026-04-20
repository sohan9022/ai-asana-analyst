# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# def train_model():
#     print("🚀 Starting MLOps Automated Training Pipeline...")
    
#     # 1. Define Paths to your specific Train/Test folders
#     train_dir = os.path.join("dataset", "train")
#     test_dir = os.path.join("dataset", "test")
    
#     img_height = 75
#     img_width = 75
#     batch_size = 32

#     # 2. Load the Training and Validation Data explicitly
#     print("Loading training dataset...")
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#         train_dir,
#         image_size=(img_height, img_width),
#         batch_size=batch_size
#     )

#     print("Loading validation dataset...")
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         test_dir,
#         image_size=(img_height, img_width),
#         batch_size=batch_size
#     )

#     # Normalize pixel values to be between 0 and 1
#     normalization_layer = tf.keras.layers.Rescaling(1./255)
#     train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#     val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#     # 3. Define the CNN Architecture 
#     print("Building model architecture...")
#     model = Sequential([
#         Input(shape=(75, 75, 3)), # Fixed the Keras Input shape warning
#         Conv2D(32, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(5, activation='softmax') # Restored to 5 classes for your 5 poses!
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # 4. Train the Model
#     print("Training model...")
#     epochs = 15 
#     history = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=epochs
#     )

#     # 5. Save the Model to your App Engine
#     save_dir = "app/engine"
#     os.makedirs(save_dir, exist_ok=True)
#     model_path = os.path.join(save_dir, "yoga_pose_model.keras")
    
#     model.save(model_path)
#     print(f"✅ Training complete! New model automatically saved to {model_path}")

# if __name__ == "__main__":
#     train_model()

# # Testing the MLOps pipeline automation!
# print("Training complete. Passing to evaluation...")

# import os
# import kagglehub
# import tensorflow as tf

# def train_model():
#     print("🚀 Starting MLOps Automated Training Pipeline...")

#     # Download dataset
#     dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")
#     print("Dataset path:", dataset_path)

#     train_dir = os.path.join(dataset_path, "train")
#     test_dir = os.path.join(dataset_path, "test")

#     img_height = 75
#     img_width = 75
#     batch_size = 32

#     print("Loading training dataset...")
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#         train_dir,
#         image_size=(img_height, img_width),
#         batch_size=batch_size
#     )

#     print("Loading validation dataset...")
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         test_dir,
#         image_size=(img_height, img_width),
#         batch_size=batch_size
#     )

#     normalization_layer = tf.keras.layers.Rescaling(1./255)
#     train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#     val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(75, 75, 3)),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(5, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     print("Training model...")
#     model.fit(train_ds, validation_data=val_ds, epochs=15)

#     save_dir = "app/engine"
#     os.makedirs(save_dir, exist_ok=True)
#     model_path = os.path.join(save_dir, "yoga_pose_model.keras")

#     model.save(model_path)
#     print(f"✅ Model saved at {model_path}")

# if __name__ == "__main__":
#     train_model()

import os
import kagglehub
import tensorflow as tf

def train_model():
    print("🚀 Starting MLOps Automated Training Pipeline...")

    # 📥 Download dataset
    dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")

    print("Dataset path:", dataset_path)
    print("Root contents:", os.listdir(dataset_path))

    # 🔍 Find actual dataset folder
    subfolders = os.listdir(dataset_path)
    actual_data_path = os.path.join(dataset_path, subfolders[0])

    print("Actual data path:", actual_data_path)
    print("Inside actual path:", os.listdir(actual_data_path))

    img_height = 75
    img_width = 75
    batch_size = 32

    # ✅ CASE 1: If dataset already has train/test folders
    if "train" in os.listdir(actual_data_path) and "test" in os.listdir(actual_data_path):
        print("✅ Using existing train/test split")

        train_dir = os.path.join(actual_data_path, "train")
        test_dir = os.path.join(actual_data_path, "test")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    # ✅ CASE 2: If dataset has only class folders (no split)
    else:
        print("⚠️ No train/test split found. Using validation_split")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            actual_data_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            actual_data_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    # 🔄 Normalize data
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 🧠 Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(75, 75, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 🚀 Train
    print("Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=15)

    # 💾 Save model
    save_dir = "app/engine"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "yoga_pose_model.keras")

    model.save(model_path)
    print(f"✅ Model saved at {model_path}")


if __name__ == "__main__":
    train_model()