import os
import sys
import kagglehub
import tensorflow as tf

def evaluate_model():
    print("📊 Evaluating the new model...")

    # 📥 Download dataset
    dataset_path = kagglehub.dataset_download("niharika41298/yoga-poses-dataset")

    print("Dataset path:", dataset_path)

    # 🔍 Find actual dataset folder
    subfolders = os.listdir(dataset_path)
    actual_data_path = os.path.join(dataset_path, subfolders[0])

    print("Actual data path:", actual_data_path)
    contents = os.listdir(actual_data_path)
    print("Contents:", contents)

    img_height = 75
    img_width = 75
    batch_size = 32

    # ✅ CASE 1: TEST folder exists
    if "TEST" in contents:
        print("Using TEST folder")
        test_dir = os.path.join(actual_data_path, "TEST")

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    elif "test" in contents:
        print("Using test folder")
        test_dir = os.path.join(actual_data_path, "test")

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    # ✅ CASE 2: No test folder → use validation split
    else:
        print("⚠️ No test folder found. Using validation split")

        test_ds = tf.keras.utils.image_dataset_from_directory(
            actual_data_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

    # 🔄 Normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # 🔥 Handle corrupted images
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

    # 📦 Load trained model
    model = tf.keras.models.load_model("app/engine/yoga_pose_model.keras")

    # 📊 Evaluate
    loss, accuracy = model.evaluate(test_ds)
    print(f"🎯 New Model Accuracy: {accuracy * 100:.2f}%")

    # 🚪 Gatekeeper logic (your original logic preserved)
    if accuracy < 0.70:
        print("❌ Model accuracy is too low. Rejecting deployment.")
        sys.exit(1)
    else:
        print("✅ Model passed evaluation! Approved for deployment.")


if __name__ == "__main__":
    evaluate_model()