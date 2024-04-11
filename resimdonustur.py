import cv2
import numpy as np

# Resmi yükle
image = cv2.imread('./testResim/WhatsApp Image 2024-01-31 at 17.15.44 (1).jpeg')
image=cv2.resize(image,(240,320))

# Resmi BGR'den RGB'ye dönüştür
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resmi HSV renk uzayına dönüştür
hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
# cv2.imshow("hsv",hsv_image)

# Genişletilmiş cilt rengi aralığını belirle
lower_skin = np.array([0, 24, 40], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Maskeleme
skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
# cv2.imshow("mask",skin_mask)

# Resimdeki sadece cilt rengini içeren alanları koruyan bir maske oluştur
skin = cv2.bitwise_and(image, image, mask=skin_mask)
# cv2.imshow("cilt",skin)

# Geri kalan kısmı griye dönüştür
gray_mask = cv2.bitwise_not(skin_mask)
gray_background = cv2.cvtColor(cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2GRAY)

# Gri arka planı oluştur
gray_background_rgb = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2RGB)

# Cilt rengi alanlarını renkli arka plan ile birleştir
result = cv2.bitwise_or(skin, gray_background_rgb)
cv2.imwrite("testResim/Y.jpg", result)
# Sonucu göster
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
