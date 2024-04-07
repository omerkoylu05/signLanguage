from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.applications import MobileNet
from keras.optimizers import Adam
import math

# Veri artırma
datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

# Eğitim ve Test setlerini yükleyin
training_set = datagen.flow_from_directory('./dataset3',
                                           target_size=(224, 224),
                                           batch_size=32,
                                           class_mode='categorical')

test_set = datagen.flow_from_directory('./dataset2',
                                       target_size=(224, 224),
                                       batch_size=32,
                                       class_mode='categorical')

# MobileNet modelini yükleyin

class_names = list(training_set.class_indices.keys())

# Write the class names to a text file
with open('class_names.txt', 'w') as f:
    f.write(str(class_names))

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Modeli oluşturun
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(units=29, activation='softmax'))  # 3, sınıf sayısına bağlıdır

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
# steps_per_epoch=29
print("spe:",steps_per_epoch)

model.fit(training_set,
          steps_per_epoch=steps_per_epoch,
          epochs=3,
          validation_data=test_set,
          validation_steps=30)

# Modeli kaydedin
model.save('./model/modelMobileNet.keras')
