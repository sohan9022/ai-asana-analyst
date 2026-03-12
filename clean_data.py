import os
from PIL import Image
import warnings

# Suppress PIL warnings about large images
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

def deep_clean_dataset(dataset_dir="dataset"):
    print("🧹 Starting Deep Clean of the dataset...")
    removed_count = 0
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                # 1. Verify the file headers
                img = Image.open(filepath)
                img.verify()
                
                # 2. To catch "premature end of data", we MUST actually load the pixels
                img = Image.open(filepath) 
                img.load() 
                
            except Exception as e:
                print(f"🚨 Corrupted file found and deleted: {filepath}")
                # Close the image before trying to delete it
                try:
                    img.close()
                except:
                    pass
                
                os.remove(filepath)
                removed_count += 1
                
    print(f"✅ Deep Clean finished! Total corrupted files removed: {removed_count}")

if __name__ == "__main__":
    deep_clean_dataset()