import numpy as np
import matplotlib.pyplot as plt

# --- 1) Build a synthetic reference image with rich texture ---
np.random.seed(0)

H, W = 128, 128
y, x = np.mgrid[0:H, 0:W]

# A textured pattern: sum of a few Gaussians + sinusoid + noise for realism
def gaussian2d(x0, y0, sx, sy, amp=1.0):
    return amp * np.exp(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))

ref = (
    gaussian2d(40, 50, 10, 8, 0.9) +
    gaussian2d(85, 80, 7, 12, 0.7) +
    0.3*np.sin(0.15*x) * np.cos(0.12*y) +
    0.15*np.random.default_rng(42).standard_normal((H, W))
)

# Normalize to [0, 255] as typical 8-bit image (but keep float)
ref = ref - ref.min()
ref = 255 * ref / ref.max()

# --- 2) Define a subpixel bilinear sampler ---
def bilinear_sample(img, coords):
    """
    img: HxW float image
    coords: Nx2 array of [y', x'] to sample (subpixel)
    returns: N samples
    """
    H, W = img.shape
    ys, xs = coords[:, 0], coords[:, 1]
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip to image bounds
    x0c = np.clip(x0, 0, W-1); x1c = np.clip(x1, 0, W-1)
    y0c = np.clip(y0, 0, H-1); y1c = np.clip(y1, 0, H-1)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    wx = xs - x0
    wy = ys - y0
    wa = (1-wx)*(1-wy)
    wb = wx*(1-wy)
    wc = (1-wx)*wy
    wd = wx*wy

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

# --- 3) Choose a subset window and generate the "deformed" image ---
cx, cy = 64, 64
half = 16  # subset radius ~16 -> 33x33 subset
win_x = np.arange(cx-half, cx+half+1)
win_y = np.arange(cy-half, cy+half+1)
grid_x, grid_y = np.meshgrid(win_x, win_y)

# True displacement and photometric change (a: contrast, b: brightness)
u_true, v_true = 2.3, -1.7
a, b = 1.8, 20.0

# Build deformed image by warping ref and applying affine photometric model
coords = np.stack([y.ravel()-v_true, x.ravel()-u_true], axis=1)
cur = bilinear_sample(ref, coords).reshape(H, W)
cur = a*cur + b  # apply photometric change

# --- Show the reference and deformed images ---
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
axs[0].imshow(ref, cmap='gray')
axs[0].set_title("Reference Image (synthetic speckle-like)")
axs[0].axis('off')
axs[1].imshow(cur, cmap='gray')
axs[1].set_title("Deformed Image (shifted + aF+b)")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# --- 4) Similarity metrics for a sweep of (u, v) around the true displacement ---
Us = np.linspace(u_true-5, u_true+5, 81)   # +/-5 px, 0.125 step
Vs = np.linspace(v_true-5, v_true+5, 81)

SSD = np.zeros((len(Vs), len(Us)))
ZNSSD = np.zeros_like(SSD)
NCC = np.zeros_like(SSD)

# Precompute reference subset vector
ref_coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=1)
F = bilinear_sample(ref, ref_coords)  # shape (N,)
F_mean = F.mean()
F_zm = F - F_mean
deltaF = np.linalg.norm(F_zm) + 1e-12
F_hat = F_zm / deltaF

for iv, v in enumerate(Vs):
    for iu, u in enumerate(Us):
        coords_uv = np.stack([grid_y.ravel()-v, grid_x.ravel()-u], axis=1)
        G = bilinear_sample(cur, coords_uv)

        # SSD
        SSD[iv, iu] = np.sum((G - F)**2)

        # ZNSSD / NCC (L2-norm variant)
        G_mean = G.mean()
        G_zm = G - G_mean
        deltaG = np.linalg.norm(G_zm) + 1e-12
        G_hat = G_zm / deltaG

        rho = float(np.dot(F_hat, G_hat))
        NCC[iv, iu] = rho
        ZNSSD[iv, iu] = 2.0*(1.0 - rho)

# --- 5) Display three separate figures (one per metric) ---
extent = [Us[0], Us[-1], Vs[-1], Vs[0]]  # axes show u (x), v (y)

def show_surface(Z, title, cbar_label):
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(Z, extent=extent, origin='upper', aspect='equal')
    plt.colorbar(label=cbar_label)
    plt.xlabel('u (pixels)')
    plt.ylabel('v (pixels)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

show_surface(SSD, 'SSD surface (lower is better)', 'SSD')
show_surface(ZNSSD, 'ZNSSD surface (lower is better)', 'ZNSSD')
show_surface(NCC, 'NCC surface (higher is better)', 'NCC')
