import pytesseract
from PIL import Image
import os
import json

#I have done this cause I did not have tesseract in environment variables.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img_folder = './images/Book_en'
results = []

for img_name in os.listdir(img_folder):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        continue
    try:    
        img_path = os.path.join(img_folder, img_name)
        text = pytesseract.image_to_string(Image.open(img_path))
        results.append({'image': img_name, 'pred': text})
    except Exception as e:
        print("❌ Error with", img_name, ":", e)
        # Save empty prediction if error occurs for robust batching
        results.append({'image': img_name, 'pred': ""})

with open('tesseract_results.json', 'w', encoding='utf-8') as f:  #will save in current directory
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Saved", len(results), "results")
