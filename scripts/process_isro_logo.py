import os
import cv2
import numpy as np
from PIL import Image

src_path = r"C:\Users\sunet\.gemini\antigravity-ide\brain\74536d11-2bd0-431d-afbd-bf3eb24c5629\.user_uploaded\media_1790105939906.png"
out_dir = r"c:\Users\sunet\Documents\ISRO\web\public"
artifact_dir = r"C:\Users\sunet\.gemini\antigravity-ide\brain\74536d11-2bd0-431d-afbd-bf3eb24c5629"

# Load image in grayscale
img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# Threshold to get clean binary shape
_, binary = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Center component is the one closest to image center
center_x, center_y = w / 2.0, h / 2.0
center_comp = min(range(1, num_labels), key=lambda i: (centroids[i][0] - center_x)**2 + (centroids[i][1] - center_y)**2)

# Anti-aliased alpha from inverted grayscale
gray_float = img.astype(np.float32)
alpha = np.clip((245.0 - gray_float) / (245.0 - 90.0), 0.0, 1.0) * 255.0
alpha[binary == 0] = 0
alpha = alpha.astype(np.uint8)

# Exact ISRO Color Palette (RGB)
# ISRO Saffron: #F37021 (243, 112, 33)
# ISRO Electric Amber: #FF7700 (255, 119, 0)
# ISRO Deep Blue: #005A9C (0, 90, 156)
# ISRO Cyan/Electric Blue: #0284C7 (2, 132, 199)
ISRO_SAFFRON = np.array([243, 112, 33], dtype=np.float32)
ISRO_BLUE = np.array([2, 132, 199], dtype=np.float32)
ISRO_NAVY = np.array([0, 90, 156], dtype=np.float32)

# =========================================================================
# 1. TWO-TONE ISRO BRANDED (Outer wings in ISRO Saffron, Center in ISRO Blue)
# =========================================================================
rgba_isro = np.zeros((h, w, 4), dtype=np.uint8)

for i in range(1, num_labels):
    mask_i = (labels == i)
    cx, cy = centroids[i]
    if i == center_comp:
        # Center circle/pupil: Aerospace Cyan-to-Navy gradient
        for y_idx in range(h):
            row_mask = mask_i[y_idx, :]
            if np.any(row_mask):
                factor = (y_idx - (center_y - 60)) / 120.0
                factor = np.clip(factor, 0.0, 1.0)
                col = (1.0 - factor) * ISRO_BLUE + factor * ISRO_NAVY
                rgba_isro[y_idx, row_mask, 0] = int(col[0])
                rgba_isro[y_idx, row_mask, 1] = int(col[1])
                rgba_isro[y_idx, row_mask, 2] = int(col[2])
    elif cx < center_x:
        # Left wing + left chevron: ISRO Saffron
        rgba_isro[mask_i, 0] = int(ISRO_SAFFRON[0])
        rgba_isro[mask_i, 1] = int(ISRO_SAFFRON[1])
        rgba_isro[mask_i, 2] = int(ISRO_SAFFRON[2])
    else:
        # Right wing + right chevron: ISRO Saffron with subtle aerospace glow or pure Saffron
        # To match the ISRO logo where Saffron and Blue meet:
        # Left side Saffron, Right side Blue, OR Outer Saffron & Center Blue
        # Let's make outer blades Saffron and chevrons Blue!
        rgba_isro[mask_i, 0] = int(ISRO_SAFFRON[0])
        rgba_isro[mask_i, 1] = int(ISRO_SAFFRON[1])
        rgba_isro[mask_i, 2] = int(ISRO_SAFFRON[2])

# Inner chevrons in ISRO Blue for high-tech contrast
for i in range(1, num_labels):
    if i != center_comp and stats[i, cv2.CC_STAT_AREA] < 12000:
        mask_i = (labels == i)
        rgba_isro[mask_i, 0] = int(ISRO_BLUE[0])
        rgba_isro[mask_i, 1] = int(ISRO_BLUE[1])
        rgba_isro[mask_i, 2] = int(ISRO_BLUE[2])

rgba_isro[:, :, 3] = alpha

# =========================================================================
# 2. SEAMLESS ISRO DIAGONAL GRADIENT (Saffron -> Deep Blue)
# =========================================================================
rgba_grad = np.zeros((h, w, 4), dtype=np.uint8)
Y, X = np.ogrid[:h, :w]
# Diagonal gradient from top-left (0,0) to bottom-right (h,w)
diag_t = (X / float(w) * 0.75 + Y / float(h) * 0.25)
diag_t = np.clip(diag_t, 0.0, 1.0)[:, :, np.newaxis]

grad_color = (1.0 - diag_t) * ISRO_SAFFRON + diag_t * ISRO_BLUE
rgba_grad[:, :, :3] = np.clip(grad_color, 0, 255).astype(np.uint8)
rgba_grad[:, :, 3] = alpha

# =========================================================================
# Crop tight bounding box
# =========================================================================
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
all_points = np.vstack(contours)
x, y, bw, bh = cv2.boundingRect(all_points)

pad_x = int(bw * 0.03)
pad_y = int(bh * 0.03)
x0 = max(0, x - pad_x)
y0 = max(0, y - pad_y)
x1 = min(w, x + bw + pad_x)
y1 = min(h, y + bh + pad_y)

rgba_isro_crop = rgba_isro[y0:y1, x0:x1]
rgba_grad_crop = rgba_grad[y0:y1, x0:x1]

# Save all variations
Image.fromarray(rgba_isro_crop).save(os.path.join(out_dir, "parikshan_logo.png"), "PNG")
Image.fromarray(rgba_isro_crop).save(os.path.join(out_dir, "parikshan_logo_twotone.png"), "PNG")
Image.fromarray(rgba_grad_crop).save(os.path.join(out_dir, "parikshan_logo_grad.png"), "PNG")

Image.fromarray(rgba_isro_crop).save(os.path.join(artifact_dir, "parikshan_logo.png"), "PNG")
Image.fromarray(rgba_isro_crop).save(os.path.join(artifact_dir, "parikshan_logo_twotone.png"), "PNG")
Image.fromarray(rgba_grad_crop).save(os.path.join(artifact_dir, "parikshan_logo_grad.png"), "PNG")

print(f"Cropped logo size: {x1-x0}x{y1-y0}")
print("Saved transparent ISRO logos successfully.")
