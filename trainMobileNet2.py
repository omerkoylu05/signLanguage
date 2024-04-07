import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ana klasör yolunu belirtin
base_dir = './dataset3'

# Veri setini eğitim ve test setlerine ayırmak için ImageDataGenerator nesnesi oluşturun
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Eğitim setini oluşturun
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Test setini oluşturun
test_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# MobileNet modelini yükleyin
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modeli özelleştirin
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax'))

# Modeli derleyin
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(train_generator, validation_data=test_generator, epochs=5)
model.save("./model/ModelNet2.keras")