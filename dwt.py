from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def haar_dwt(image):
    h, w = image.shape
    temp = np.zeros_like(image)

    for i in range(h):
        for j in range(0, w // 2):
            temp[i, j]       = (image[i, 2*j] + image[i, 2*j + 1]) / 2
            temp[i, j + w//2] = (image[i, 2*j] - image[i, 2*j + 1]) / 2

    output = np.zeros_like(image)
    for j in range(w):
        for i in range(0, h // 2):
            output[i, j]       = (temp[2*i, j] + temp[2*i + 1, j]) / 2
            output[i + h//2, j] = (temp[2*i, j] - temp[2*i + 1, j]) / 2

    return output

def haar_revert_dwt(transformed_image):
    h, w = transformed_image.shape
    LL = transformed_image[:h//2, :w//2]
    LH = transformed_image[:h//2, w//2:]
    HL = transformed_image[h//2:, :w//2]
    HH = transformed_image[h//2:, w//2:]

    temp = np.zeros((h, w), dtype=np.float32)
    for j in range(w//2):
        for i in range(h//2):
            temp[2*i, j]     = LL[i, j] + HL[i, j]
            temp[2*i + 1, j] = LL[i, j] - HL[i, j]
    for j in range(w//2, w):
        for i in range(h//2):
            temp[2*i, j]     = LH[i, j - w//2] + HH[i, j - w//2]
            temp[2*i + 1, j] = LH[i, j - w//2] - HH[i, j - w//2]

    output = np.zeros_like(transformed_image, dtype=np.float32)
    for i in range(h):
        for j in range(w//2):
            output[i, 2*j]     = temp[i, j] + temp[i, j + w//2]
            output[i, 2*j + 1] = temp[i, j] - temp[i, j + w//2]
        
    return output

def deadzone_quantization(coeffs, delta=10, threshold=5):
    q = np.sign(coeffs) * np.maximum(0, np.floor((np.abs(coeffs) - threshold) / delta + 0.5))
    return q

def deadzone_dequantization(q, delta=10, threshold=5):
    x_hat = np.sign(q) * (np.abs(q) * delta + threshold / 2)
    return x_hat

def ssim(img1, img2, K1=0.01, K2=0.03, L=255, window_size=11, sigma=1.5):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)

    sigma1_sq = gaussian_filter(img1**2, sigma) - mu1**2
    sigma2_sq = gaussian_filter(img2**2, sigma) - mu2**2
    sigma12 = gaussian_filter(img1*img2, sigma) - mu1*mu2

    C1 = (K1*L)**2
    C2 = (K2*L)**2

    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

images = ["Type-B/EarthMoon-browse.jpg", "Type-A/kodim03.png"]

for img_path in images:
    image = Image.open(img_path).convert("L")
    img_array = np.array(image).astype(np.float32)

    dwt_result = haar_dwt(img_array)

    h, w = dwt_result.shape
    LL = dwt_result[:h//2, :w//2]
    LH = dwt_result[:h//2, w//2:]
    HL = dwt_result[h//2:, :w//2]
    HH = dwt_result[h//2:, w//2:]

    fig, axs = plt.subplots(1, 5, figsize=(16, 4))
    axs[0].imshow(img_array, cmap='gray')
    axs[0].set_title("Image originale")
    axs[0].axis('off')

    axs[1].imshow(LL, cmap='gray')
    axs[1].set_title("LL (basses fréquences)")
    axs[1].axis('off')

    axs[2].imshow(LH, cmap='gray')
    axs[2].set_title("LH (détails horizontaux)")
    axs[2].axis('off')

    axs[3].imshow(HL, cmap='gray')
    axs[3].set_title("HL (détails verticaux)")
    axs[3].axis('off')

    axs[4].imshow(HH, cmap='gray')
    axs[4].set_title("HH (détails diagonaux)")
    axs[4].axis('off')

    plt.suptitle(f"Décomposition DWT (Haar) : {img_path}")
    plt.tight_layout()
    plt.show()
    print(dwt_result)

    q_dwt = deadzone_quantization(dwt_result, delta=5, threshold=15)
    dq_dwt = deadzone_dequantization(q_dwt, delta=5, threshold=15)

    reconstructed_image = haar_revert_dwt(dq_dwt)

    mse = np.mean((img_array - reconstructed_image) ** 2)
    mae = np.mean(np.abs(img_array - reconstructed_image))
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    score = ssim(img_array, reconstructed_image)

    print(f"➡ {img_path}")
    print(f"  MAE : {mae:.3f}")
    print(f"  MSE : {mse:.3f}")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM: {score:.3f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Image reconstruite (IDWT)")
    plt.axis('off')

    plt.suptitle(f"Comparaison DWT/IDWT - {img_path}")
    plt.show()
