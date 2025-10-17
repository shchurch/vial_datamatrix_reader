from pylibdmtx import pylibdmtx
from PIL import Image
import pandas as pd
import cv2
import numpy as np

IMAGE_PATH = "input.jpg"
OUTPUT_IMAGE = "annotated_output.jpg"
TOLERANCE = 30  # row tolerance for grid ordering

# Load image
img_pil = Image.open(IMAGE_PATH)
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Decode DataMatrix codes
decoded = pylibdmtx.decode(img_pil)

records = []
for d in decoded:
    try:
        val = d.data.decode('ascii')  # letters and numbers
    except UnicodeDecodeError:
        val = d.data.hex()  # fallback
    
    x, y, w, h = d.rect.left, d.rect.top, d.rect.width, d.rect.height
    cx, cy = x + w/2, y + h/2
    records.append({"value": val, "x": cx, "y": cy, "rect": (x, y, w, h)})

# Convert to DataFrame
df = pd.DataFrame(records)

if df.empty:
    print("No DataMatrix codes detected.")
else:
    # Sort top-left → right → down
    df = df.sort_values(by=["y", "x"]).reset_index(drop=True)
    
    # Assign approximate grid row/col
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
    df.to_csv("datamatrix_table.csv", index=False)
    print(df[['row', 'col', 'value']])
    print(f"Annotated image saved as: {OUTPUT_IMAGE}")
