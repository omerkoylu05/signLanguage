from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# Load the pre-trained ResNet50 model, excluding the top layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Add global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a dense layer
x = Dense(1024, activation='relu')(x)

# Add a final output layer for your number of classes (replace 'num_classes' with your actual number of classes)
predictions = Dense(29, activation='softmax')(x)

# Create the full network so we can train on it
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator to preprocess your images
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory('./dataset3',
                                                 target_size=(48, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')
print(training_set.class_indices)

class_names = list(training_set.class_indices.keys())

# Write the class names to a text file
with open('class_names.txt', 'w') as f:
    f.write(str(class_names))
# Train the model
steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
# steps_per_epoch=29
print("steps:",steps_per_epoch)
model.fit(training_set, steps_per_epoch=steps_per_epoch, epochs=5)
model.save("./model/model.h5")