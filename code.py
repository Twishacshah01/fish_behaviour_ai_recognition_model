import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the image data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
train_generator = train_datagen.flow_from_directory(
    r'C:\\Users\\twish\\Downloads\\Aquarium Dataset.v10i.yolov5pytorch',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    r'C:\\Users\\twish\\Downloads\\Aquarium Dataset.v10i.yolov5pytorch',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_generator.class_indices)))  # Output layer with number of classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)




#evaluate
loss, accuracy = model.evaluate(val_generator)
print(f'Validation loss: {loss:.2f}')
print(f'Validation accuracy: {accuracy:.2f}')



model.save('image_classification_model.h5')
from tensorflow.keras.models import load_model

model = load_model('image_classification_model.h5')

predictions = model.predict(val_generator)

loss, accuracy = model.evaluate(val_generator)
print(f'Test loss: {loss:.2f}')
print(f'Test accuracy: {accuracy:.2f}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get the true labels
true_labels = val_generator.classes

# Get the predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the image
img_path =  r'C:\Users\twish\output_frames\frame_832.jpg'
img = load_img(img_path, target_size=(224, 224))

# Convert the image to an array
img_array = img_to_array(img)

# Expand the dimensions
img_array = np.expand_dims(img_array, axis=0)

# Make predictions
predictions = model.predict(img_array)

# Get the predicted label
predicted_label = np.argmax(predictions, axis=1)

print(f'Predicted label: {predicted_label}')



import matplotlib.pyplot as plt

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


import cv2

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    # Process the frame (e.g., detect objects)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
