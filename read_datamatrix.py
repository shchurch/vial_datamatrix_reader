# read_datamatrix.py
import sys
from pylibdmtx import pylibdmtx
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# === CONFIG ===
TOLERANCE = 100          # row/column grouping
MAX_DIM = 100000          # max width/height for detection (resizing)
DEBUG = False           # set True to see resized image overlay

# === PARSE ARGUMENTS ===
if len(sys.argv) < 4:
    print("Usage: python read_datamatrix.py <image_path> <output_image> <output_csv>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
OUTPUT_IMAGE = sys.argv[2]
OUTPUT_CSV = sys.argv[3]

# === LOAD IMAGE ===
img_pil = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = img_pil.width, img_pil.height

# === RESIZE IMAGE FOR FASTER DETECTION ===
scale = min(MAX_DIM / orig_w, MAX_DIM / orig_h, 1.0)
if scale < 1.0:
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img_pil_small = img_pil.resize((new_w, new_h), Image.LANCZOS)
else:
    img_pil_small = img_pil

# === CONVERT TO GRAYSCALE ===
img_gray = img_pil_small.convert("L")

# === DECODE DATAMATRIX ===
decoded = pylibdmtx.decode(img_gray)

# === PREPARE ORIGINAL IMAGE FOR OVERLAY ===
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
height, width = img_cv.shape[:2]

records = []
for d in decoded:
    try:
        val = d.data.decode("ascii")
    except UnicodeDecodeError:
        val = d.data.hex()  # fallback if non-ASCII
    
    # Scale coordinates back to original image
    x = int(d.rect.left / scale)
    y = int(d.rect.top / scale)
    w = int(d.rect.width / scale)
    h = int(d.rect.height / scale)
    
    # Optional vertical flip if overlay appears upside down
    #y_corrected = y
    y_corrected = height - (y + h)
    
    cx, cy = x + w/2, y_corrected + h/2
    records.append({"value": val, "x": cx, "y": cy, "rect": (x, y_corrected, w, h)})

# === CREATE DATAFRAME ===
df = pd.DataFrame(records)
if df.empty:
    print("No DataMatrix codes detected.")
else:
    df['row'] = (df['y'] // TOLERANCE).astype(int)
    df['col'] = (df['x'] // TOLERANCE).astype(int)
    df = df.sort_values(by=["row", "col"]).reset_index(drop=True)

    # Draw rectangles and labels
    for idx, row in df.iterrows():
        x, y, w, h = row['rect']
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_cv,
            f"{idx+1}: {row['value']}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    # Save outputs
    cv2.imwrite(OUTPUT_IMAGE, img_cv)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Detection complete.")
    print(f"Annotated image saved to: {OUTPUT_IMAGE}")
    print(f"CSV table saved to: {OUTPUT_CSV}")
    print(df[['row', 'col', 'value']])

# === OPTIONAL DEBUG ===
if DEBUG:
    # Show resized image with rectangles (for testing)
    import matplotlib.pyplot as plt
    img_small_cv = cv2.cvtColor(np.array(img_pil_small), cv2.COLOR_RGB2BGR)
    for row in records:
        x, y, w, h = int(row['x']*scale), int(row['y']*scale), int(row['rect'][2]*scale), int(row['rect'][3]*scale)
        cv2.rectangle(img_small_cv, (x, y), (x + w, y + h), (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(img_small_cv, cv2.COLOR_BGR2RGB))
    plt.show()
