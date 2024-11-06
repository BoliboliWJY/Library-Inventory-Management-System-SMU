#%%
#注意，这个跑的特别慢，且优化效果不明显！未经过并行优化，加上代码估计有问题
import numpy as np
import matplotlib.pyplot as plt

import os#修改当前路径
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())

#%%
#获取文件数据
import struct
def read_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

images_file = os.path.join(script_dir, r'archive\train-images.idx3-ubyte')
labels_file = os.path.join(script_dir, r'archive\train-labels.idx1-ubyte')

images = read_images(images_file)
labels = read_labels(labels_file)

# images = images[0:10000]
# labels = labels[0:10000]

# %%
#打印原始数据
print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')
show_row, show_col = 2, 5
plt.figure(figsize=(show_col*2, show_row*2))
for i in range(show_row*show_col):
    plt.subplot(show_row, show_col, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.axis('off')
# %%
#定义各种层
class FullyConnectedLayer:#全连接层
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.bias = np.zeros(output_len)
    
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
    
    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_weights = np.dot(self.input.T, d_L_d_out) / self.input.shape[0]
        d_L_d_bias = np.mean(d_L_d_out, axis=0)
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        self.weights -= learn_rate * d_L_d_weights
        self.bias -= learn_rate * d_L_d_bias
        return d_L_d_input

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):#relu的导数
    return np.where(x>0,1,0)#等价于x>0?1:0

class ConvLayer:#卷积层
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size+1):
            for j in range(w - self.filter_size + 1):
                img_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield img_region, i, j
                
    def iterate_pooled_regions(self, image):
        h, w = image.shape
        for i in range(pool_output_width):
            for j in range(pool_output_height):
                img_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield img_region, i, j
            
    def forward(self, input):
        self.last_input = input
        batch_size, h, w = input.shape
        output = np.zeros((batch_size, h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        for b in range(batch_size):
            for img_region, i, j in self.iterate_regions(input[b]):#二维扫描
                output[b, i, j] = np.sum(img_region * self.filters, axis = (1, 2))
        return relu(output)
    
    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for b in range(self.last_input.shape[0]):
            for img_region, i, j in self.iterate_pooled_regions(self.last_input[b]):
                for f in range(self.num_filters):
                    if d_L_d_out[b,i,j,f] <= 0:
                        continue
                    d_L_d_filters[f] += d_L_d_out[b,i,j,f]*img_region
        self.filters -= learn_rate * d_L_d_filters
        return None
    
    
def max_pooling(input, size, stride):#池化层
    #格子大小，步长
    batch_size, h, w, num_filters = input.shape
    new_h = (h - size) // stride + 1
    new_w = (w - size) // stride + 1
    pooled = np.zeros((batch_size, new_h, new_w, num_filters))
    
    for b in range(batch_size):
        for i in range(0,h,stride):
            for j in range(0,w,stride):
                img_region = input[b,i:i+size,j:j+size]
                pooled[b,i//stride, j//stride] = np.max(img_region,axis=(0,1))#采用最大池化层
    return pooled

def softmax(x):#输出层
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    delta = 1e-7
    return -np.sum(np.log(probs[np.arange(len(labels)), labels] + delta)) / len(labels)

def calculate_output_size(input_size, filter_size, stride):
    return (input_size - filter_size) // stride + 1

conv = ConvLayer(num_filters=8, filter_size=3)#每层卷积核数量，卷积核大小
pool_size = 2
input_height, input_width = images.shape[1], images.shape[2]
conv_output_height = calculate_output_size(input_height, conv.filter_size, 1)
conv_output_width = calculate_output_size(input_width, conv.filter_size, 1)
pool_output_height = calculate_output_size(conv_output_height, pool_size, pool_size)
pool_output_width = calculate_output_size(conv_output_width, pool_size, pool_size)
output_size_after_pooling = pool_output_height * pool_output_width * conv.num_filters
fc = FullyConnectedLayer(input_len = output_size_after_pooling, output_len = 20)
learn_rate = 0.005
epochs = 50
batch_size = 100
losses = []

for epoch in range(epochs):
    print(f'--- Epoch {epoch+1} ---')
    epoch_losses = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        conv_out = conv.forward(batch_images)
        pool_out = max_pooling(conv_out, pool_size, pool_size)
        flat = pool_out.reshape(batch_size, -1)
        fc_out = fc.forward(flat)
        probs = softmax(fc_out)
        #loss
        loss = cross_entropy_loss(probs, batch_labels)
        epoch_losses.append(loss)
        print(f'Batch {i//batch_size+1}, Loss: {loss}')
        #反向
        d_L_d_out = probs
        d_L_d_out[np.arange(batch_size), batch_labels] -= 1
        d_L_d_out /= batch_size
        
        d_L_d_fc = fc.backprop(d_L_d_out, learn_rate)
        d_L_d_pool = d_L_d_fc.reshape(batch_size, pool_out.shape[1], pool_out.shape[2], conv.num_filters)
        conv.backprop(d_L_d_pool, learn_rate)
        
    average_loss = np.mean(epoch_losses)
    losses.append(average_loss)
    print(f'Epoch {epoch+1}, Average Loss: {average_loss}')
        
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()     