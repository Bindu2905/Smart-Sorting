# train_model.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ 1. Paths
train_dir = "dataset/train"
test_dir = "dataset/test"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# ✅ 2. Image generators (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# ✅ 3. Load images automatically
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print("✅ Classes detected:", train_generator.class_indices)

# ✅ 4. Build model using MobileNetV2 (transfer learning)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base layers for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ✅ 5. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ 6. Train model
EPOCHS = 10
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ✅ 7. Fine-tune (unfreeze last few layers of base model)
base_model.trainable = True
for layer in base_model.layers[:-30]:  # unfreeze top 30 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🔁 Fine-tuning the model...")
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# ✅ 8. Save model and class names
model.save(os.path.join(model_dir, "fruit_model.h5"))

# Save class indices for later use in app.py
with open(os.path.join(model_dir, "classes.txt"), "w") as f:
    for label, idx in train_generator.class_indices.items():
        f.write(f"{idx}:{label}\n")

print("🎉 Training complete! Model saved to model/fruit_model.h5")