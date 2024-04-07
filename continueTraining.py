from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# Load the previously trained model
model = load_model('./model/model.h5')

# Use ImageDataGenerator to preprocess your new training images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('./dataset3',
                                                     target_size=(48, 64),
                                                     batch_size=32,
                                                     class_mode='categorical')

# Continue training the model
steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
model.fit_generator(training_set, steps_per_epoch=steps_per_epoch, epochs=5)
model.save("./model/model.h5")
