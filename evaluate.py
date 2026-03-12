import tensorflow as tf
import sys

def evaluate_model():
    print("📊 Evaluating the new model...")
    
    # 1. Load your test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "dataset/test", image_size=(75, 75), batch_size=32
    )
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # 2. Load the newly trained model
    model = tf.keras.models.load_model("app/engine/yoga_pose_model.keras")

    # 3. Test its accuracy
    loss, accuracy = model.evaluate(test_ds)
    print(f"🎯 New Model Accuracy: {accuracy * 100:.2f}%")

    # 4. The Gatekeeper Logic
    if accuracy < 0.70: # If accuracy is below 70%, fail the pipeline!
        print("❌ Model accuracy is too low. Rejecting deployment.")
        sys.exit(1) # This command physically stops the GitHub Action
    else:
        print("✅ Model passed evaluation! Approved for deployment.")

if __name__ == "__main__":
    evaluate_model()