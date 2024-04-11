import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ana klasör yolunu belirtin
base_dir = './dataset3'

# Veri setini eğitim ve test setlerine ayırmak için ImageDataGenerator nesnesi oluşturun
datagen = ImageDataGenerator(rescale=1./255)

# Validation setini oluşturun
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(240, 320),  # Eğer modeliniz farklı bir input size bekliyorsa burayı güncelleyin
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

# Kayıtlı modeli yükleyin
model = tf.keras.models.load_model('./model/model_3_0.95.keras')

# Modelin accuracy değerini ölçün
loss, accuracy = model.evaluate(validation_generator)
print('Accuracy: %.2f' % (accuracy*100))
