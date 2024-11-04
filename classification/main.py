import os
import keras
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())
 
model = ResNet50(weights=None)
model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')# 使用离线权重，用于脱机运行

for i in range(1, 11):
    img_path = r'data\rust.jpg'
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    # plt.imshow(img) #可视化一下
    # plt.axis('off')
    # plt.show()
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])