from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = Path("dataset/chest_xray/train")

normal_folder = dataset_path / "NORMAL"
pneumonia_folder = dataset_path / "PNEUMONIA"

normal_images = list(normal_folder.glob("*"))
pneumonia_images = list(pneumonia_folder.glob("*"))

print("Normal images:", len(normal_images))
print("Pneumonia images:", len(pneumonia_images))

# Show Normal image
print("Showing Normal image...")
normal_image = Image.open(normal_images[0]).convert("L")
plt.figure(figsize=(5, 5))
plt.imshow(normal_image, cmap="gray")
plt.title("Normal Chest X-ray")
plt.axis("off")
plt.show()

# Show Pneumonia image
print("Showing Pneumonia image...")
pneumonia_image = Image.open(pneumonia_images[0]).convert("L")
plt.figure(figsize=(5, 5))
plt.imshow(pneumonia_image, cmap="gray")
plt.title("Pneumonia Chest X-ray")
plt.axis("off")
plt.show()

print("Done!")