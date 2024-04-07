from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# Create the model
model = Sequential()

# Add layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(29, activation='softmax'))  # We have 26 classes (letters A-Z)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator to preprocess our images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('./dataset3',
                                                 target_size=(48, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
print("spe",steps_per_epoch)

# Train the model
model.fit_generator(training_set, steps_per_epoch=steps_per_epoch,epochs=5)
model.save("./model/model.h5")