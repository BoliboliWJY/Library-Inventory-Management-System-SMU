#%%
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())
model_weights_path = r'.\model\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#%%
num_classes = 2

model = Sequential()
model.add(ResNet50(include_top=False, pooling = 'avg', weights = model_weights_path))
model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable = False
model.summary()
# %%
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# %%
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_size = 224
current_path = os.getcwd()
print("当前工作目录:", current_path)
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        r'.\input\train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        r'.\input\val',
        target_size=(image_size, image_size),
        batch_size=20,
        class_mode='categorical')
# %%
import matplotlib.pyplot as plt

# 获取训练历史
history = model.fit(
        train_generator,
        steps_per_epoch=2,
        validation_data=validation_generator,
        validation_steps=1,
        epochs = 3)

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# 绘制训练 & 验证的准确率
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

# %%
model.save('transfer_model.keras')
# %%
