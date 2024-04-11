from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Modeli yükleyin
model = load_model('./model/model_3_0.95.keras')

# Tahmin yapılacak resimlerin bulunduğu klasör
image_dir = './testResim'

# Sınıf isimlerinizi buraya koyun
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','hastane','i', 'j', 'k', 'l', 'm','manifest', 'n', 'o', 'p','polis', 'r', 's','saat', 't', 'u', 'v', 'y', 'z', 'ç', 'ö', 'ü', 'ğ', 'ı', 'ş']  # Toplamda 33 sınıf olduğunu belirtmiştiniz.
print(len(class_names))
# Resimlerin bulunduğu klasördeki her dosya için döngü
s=0
for file_name in os.listdir(image_dir):
    # Resmi yükleyin ve yeniden boyutlandırın
    img = image.load_img(os.path.join(image_dir, file_name), target_size=(240, 320))
    # s+=1
    # img.save("./tahminResim/"+str(s)+".jpg")

    # Resmi bir numpy dizisine dönüştürün
    x = image.img_to_array(img)
    # print(x.shape)

    # Resmi bir batch'e dönüştürün (Resimlerin toplu işlem boyutunu ekleyin)
    x = np.expand_dims(x, axis=0)

    # Veriyi ön işleme yapın (örneğin, [0, 1] aralığına ölçeklendirme)
    x /= 255.

    # Resim için tahminler yapın
    preds = model.predict(x,verbose=0)

    # En yüksek olasılıklı tahminin indeksini alın
    top_pred = np.argmax(preds[0])
    if (preds[0][top_pred] * 100>60):
        # En yüksek olasılıklı tahminin sınıf adını ve yüzdesini yazdırın
        print(f"Resim: {file_name} - Tahmin: {class_names[top_pred]} - Olasılık: {preds[0][top_pred] * 100}%")
