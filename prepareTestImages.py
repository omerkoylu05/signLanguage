import cv2
import os

# Fotoğrafların bulunduğu klasör
input_dir = "./testPrepDir"

# Yeni oluşturulacak klasör
output_dir = "./testResim"

# Yeni fotoğraf boyutları
width = 240
height = 320

# Verilen klasördeki tüm fotoğrafları işler
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(input_dir, filename))

        # Orta bölgeyi alır
        start_x = img.shape[1]//2 - width//2
        start_y = img.shape[0]//2 - height//2
        cropped_img = img[start_y:start_y+height, start_x:start_x+width]

        # Yeni fotoğrafı kaydeder
        cv2.imwrite(os.path.join(output_dir, filename), cropped_img)
