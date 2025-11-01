from PIL import Image
import numpy as np
import cv2

images = ["Type-B/EarthMoon-browse.jpg", "Type-A/kodim03.png"]
for img_path in images:
    image = Image.open(img_path).convert("L")
    img_array = np.array(image).astype(np.float32)

    dct = cv2.dct(img_array)

    n_colors = len(np.unique(img_array.astype(np.uint8)))
    laplacian_var = cv2.Laplacian(img_array.astype(np.float64), cv2.CV_64F).var()


    print(f"Image: {img_path}")
    print("  Nombre de couleurs :", n_colors)
    print("  Variance du Laplacien :", laplacian_var)

    h, w = img_array.shape
    low = np.mean(np.abs(dct[:h//4, :w//4]))
    mid = np.mean(np.abs(dct[h//4:h//2, w//4:w//2]))
    high = np.mean(np.abs(dct[h//2:, w//2:]))
    total_energy = low + mid + high
    low_ratio  = low / total_energy
    mid_ratio  = mid / total_energy
    high_ratio = high / total_energy
    print("  Énergie basse fréquence (DCT) :", low_ratio)
    print("  Énergie moyenne fréquence (DCT) :", mid_ratio)
    print("  Énergie haute fréquence (DCT) :", high_ratio)
