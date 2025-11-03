from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def dct_matrix(N = 8):
    A = np.zeros((N, N), dtype=np.float64)
    for u in range(N):
        alpha = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
        for x in range(N):
            A[u, x] = alpha * np.cos(np.pi * (2 * x + 1) * u / (2.0 * N))
    return A

A8 = dct_matrix(8)
A8T = A8.T

def dct2_block(b8):
    return A8 @ b8 @ A8T

def idct2_block(C8):
    return A8T @ C8 @ A8

def deadzone_quant(x, delta = 6.0, T = 12.0):
    return np.sign(x) * np.maximum(0.0, np.floor((np.abs(x) - T) / delta + 0.5))

def deadzone_dequant(q, delta = 6.0, T = 12.0):
    return np.sign(q) * (np.abs(q) * delta + T / 2.0)

def pad_to_multiple(img, blk = 8):
    H, W = img.shape
    H2 = (H + blk - 1) // blk * blk
    W2 = (W + blk - 1) // blk * blk
    out = np.zeros((H2, W2), dtype = np.float64)
    out[:H, :W] = img
    return out, H, W

def iterate_blocks(img, blk = 8):
    H, W = img.shape
    for i in range(0, H, blk):
        for j in range(0, W, blk):
            yield i, j, img[i:i + blk, j:j + blk]

def place_block(dst, i, j, b):
    dst[i:i + 8, j:j + 8] = b

def ssim_gray(img1, img2, K1 = 0.01, K2 = 0.03, L = 255.0, sigma = 1.5):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, sigma)
    mu2 = gaussian_filter(img2, sigma)

    sigma1_sq = gaussian_filter(img1**2, sigma) - mu1**2
    sigma2_sq = gaussian_filter(img2**2, sigma) - mu2**2
    sigma12   = gaussian_filter(img1*img2, sigma) - mu1*mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

images = ["Type-B/EarthMoon-browse.jpg", "Type-A/kodim03.png"]

delta, T = 6.0, 12.0

for path in images:
    img = Image.open(path).convert("L")
    f0 = np.array(img, dtype=np.float64)

    f, H, W = pad_to_multiple(f0, blk=8)

    C = np.zeros_like(f)
    for i, j, b in iterate_blocks(f, blk = 8):
        Cb = dct2_block(b)
        place_block(C, i, j, Cb)

    Q = np.zeros_like(C)
    for i, j, Cb in iterate_blocks(C, blk = 8):
        qb = deadzone_quant(Cb, delta=delta, T=T)
        place_block(Q, i, j, qb)

    frec = np.zeros_like(f)
    for i, j, qb in iterate_blocks(Q, blk = 8):
        Cb_hat = deadzone_dequant(qb, delta=delta, T=T)
        b_hat  = idct2_block(Cb_hat)
        place_block(frec, i, j, b_hat)

    frec = np.clip(frec[:H, :W], 0, 255)

    eps = 1e-9
    C_vis  = np.log1p(np.abs(C[:H, :W]))
    Q_vis  = np.log1p(np.abs(Q[:H, :W]))

    plt.figure(figsize=(14,8))
    plt.subplot(2,2,1); plt.imshow(f0, cmap="gray"); plt.title("Image originale"); plt.axis("off")
    plt.subplot(2,2,2); plt.imshow(C_vis, cmap="gray"); plt.title("Carte des coefficients DCT (log)"); plt.axis("off")
    plt.subplot(2,2,3); plt.imshow(Q_vis, cmap="gray"); plt.title("Après quantification (log)"); plt.axis("off")
    plt.subplot(2,2,4); plt.imshow(frec, cmap="gray"); plt.title("Image reconstruite (IDCT)"); plt.axis("off")
    plt.suptitle(f"DCT blocs 8x8 — {path} — Δ={delta}, T={T}")
    plt.tight_layout()
    plt.show()

    mae  = np.mean(np.abs(f0 - frec))
    mse  = np.mean((f0 - frec) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    score = ssim_gray(f0, frec)

    print(f"➡ {path}")
    print(f"  MAE : {mae:.3f}")
    print(f"  MSE : {mse:.3f}")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM: {score:.3f}")
