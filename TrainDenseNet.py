import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ana klasör yolunu belirtin
base_dir = './dataset3'

# Veri setini rastgele karıştırarak ve sonra eğitim ve test setlerine ayırmak için ImageDataGenerator nesnesi oluşturun
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Eğitim setini oluşturun
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(240, 320),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Test setini oluşturun
test_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(240, 320),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# DenseNet modelini yükleyin
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(240, 320, 3))

# Modeli özelleştirin
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax'))

# Modeli derleyin
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(train_generator, validation_data=test_generator, epochs=10)
model.save("DenseNet1.keras")