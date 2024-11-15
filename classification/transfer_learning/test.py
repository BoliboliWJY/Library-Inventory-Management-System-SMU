from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())

# 加载模型
model = load_model('transfer_model.keras')

# 预测新图像
img_path = 'rust.jpg'
img = image.load_img(img_path, target_size=(500, 500))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class}")