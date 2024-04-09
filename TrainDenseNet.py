import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Ana klasör yolunu belirtin
base_dir = './dataset3'

# Veri setini rastgele karıştırarak ve sonra eğitim ve test setlerine ayırmak için ImageDataGenerator nesnesi oluşturun
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Eğitim setini oluşturun
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(240, 320),
    batch_size=12,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Test setini oluşturun
test_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(240, 320),
    batch_size=12,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# DenseNet modelini yükleyin
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(240, 320, 3))

# # Modeli özelleştirin
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax'))
opt=Adam(learning_rate=0.0001)
# Modeli derleyin
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model=load_model("./model/model_6_0.35.keras")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('./model/model_{epoch}_{val_accuracy:.2f}.keras', monitor='val_accuracy', save_best_only=False, save_weights_only=False)

# Modeli eğitin
model.fit(train_generator, validation_data=test_generator, callbacks=[early_stopping,model_checkpoint], epochs=10)
model.save("DenseNet1.keras")