import os
import cv2

# Ana klasör yolunu belirtin
base_dir = './dataset3'

# Ana klasördeki tüm alt klasörleri ve dosyaları dolaş
for root, dirs, files in os.walk(base_dir):
    # Her dosya için
    for file in files:
        # Dosyanın tam yolunu al
        file_path = os.path.join(root, file)
        # Dosyanın bir resim olup olmadığını kontrol et (sadece .jpg ve .png dosyaları)
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            # Resmi oku
            img = cv2.imread(file_path)
            # Resmi belirttiğiniz boyutlara yeniden boyutlandır
            img = cv2.resize(img, (480, 640))
            # Resmi aynı dosyaya yaz
            cv2.imwrite(file_path, img)
