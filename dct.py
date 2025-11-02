from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def dct_matrix(N):
    A = np.zeros((N, N), dtype=np.float64)
    for u in range(N):
        alpha = np.sqrt(1.0/N) if u == 0 else np.sqrt(2.0/N)
        for x in range(N):
            A[u, x] = alpha * np.cos(np.pi * (2*x + 1) * u / (2.0 * N))
    return A

def dct2(image):
    # image: (H, W) float
    H, W = image.shape
    Ah = dct_matrix(H)
    Aw = dct_matrix(W)
    return Ah @ image @ Aw.T

def idct2(coeffs):
    H, W = coeffs.shape
    Ah = dct_matrix(H)
    Aw = dct_matrix(W)
    return Ah.T @ coeffs @ Aw

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

def split_quadrants_dct(C):
    H, W = C.shape
    h2, w2 = H//2, W//2
    LL = C[:h2, :w2]
    LH = C[:h2, w2:]
    HL = C[h2:, :w2]
    HH = C[h2:, w2:]
    return LL, LH, HL, HH

images = ["Type-B/EarthMoon-browse.jpg", "Type-A/kodim03.png"]

for img_path in images:
    image = Image.open(img_path).convert("L")
    img_array = np.array(image).astype(np.float64)

    C = dct2(img_array)

    LL, LH, HL, HH = split_quadrants_dct(C)

    fig, axs = plt.subplots(1, 5, figsize=(16, 4))
    axs[0].imshow(img_array, cmap='gray')
    axs[0].set_title("Image originale")
    axs[0].axis('off')

    axs[1].imshow(np.log1p(np.abs(LL)), cmap='gray')
    axs[1].set_title("LL (basses fréquences)")
    axs[1].axis('off')

    axs[2].imshow(np.log1p(np.abs(LH)), cmap='gray')
    axs[2].set_title("LH (détails horizontaux)")
    axs[2].axis('off')

    axs[3].imshow(np.log1p(np.abs(HL)), cmap='gray')
    axs[3].set_title("HL (détails verticaux)")
    axs[3].axis('off')

    axs[4].imshow(np.log1p(np.abs(HH)), cmap='gray')
    axs[4].set_title("HH (détails diagonaux)")
    axs[4].axis('off')

    plt.suptitle(f"Décomposition DCT (2D) : {img_path}")
    plt.tight_layout()
    plt.show()

    # Ajuster delta/threshold par quadrant identique à dwt.py pour la comparaison.
    q_dct = deadzone_quantization(C, delta=5, threshold=15)
    dq_dct = deadzone_dequantization(q_dct, delta=5, threshold=15)

    recon = idct2(dq_dct)

    recon_clipped = np.clip(recon, 0, 255)

    mse = np.mean((img_array - recon_clipped) ** 2)
    mae = np.mean(np.abs(img_array - recon_clipped))
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    score = ssim(img_array, recon_clipped)

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
    plt.imshow(recon_clipped, cmap='gray')
    plt.title("Image reconstruite (IDCT)")
    plt.axis('off')

    plt.suptitle(f"Comparaison DCT/IDCT - {img_path}")
    plt.show()
