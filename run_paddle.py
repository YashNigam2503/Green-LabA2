import os, json
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

ocr = PaddleOCR(use_textline_orientation=True, lang='en')
#ocr = PaddleOCR(use_textline_orientation=True, lang='ch') # For Chinese
#ocr = PaddleOCR(use_textline_orientation=True, lang='ar') # For Arabic

img_folder = './images/Book_en'  # Change to your image folder
results = []

for img_name in os.listdir(img_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_folder, img_name)
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)
        ocr_result = ocr.predict(img_np)
        text_lines = ocr_result[0]['rec_texts']  # This gets the list of recognized text strings
        results.append({'image': img_name, 'pred': "\n".join(text_lines)})
    except Exception as e:
        print("❌ Error with", img_name, ":", e)
        # Save empty prediction if error occurs for robust batching
        results.append({'image': img_name, 'pred': ""})    

with open('paddleocr_results.json', 'w', encoding='utf-8') as f: #will save in current directory
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Saved", len(results), "results")
