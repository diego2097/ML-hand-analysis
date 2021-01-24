import os
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow import keras

def create_image_array(x_max,y_max,data):
    # Zero pad images and obtain an array to work upon
    data_final = []
    for i in data:
        left_pad = int((x_max - i.shape[0])/2)
        right_pad = x_max - i.shape[0] - int((x_max - i.shape[0])/2)
        top_pad = int((y_max - i.shape[1])/2)
        bottom_pad = y_max - i.shape[1] - int((y_max - i.shape[1])/2)
        data_final.append(np.pad(i , pad_width = ((left_pad,right_pad),(top_pad,bottom_pad),(0,0)),mode = 'constant',constant_values = 0))
    data_final = np.array(data_final)
    return(data_final)


def get_maximum_dimensions(image_paths):
    # Get list of images from the directory
    #image_paths = archive.namelist()[0:10]
    #image_paths = filter(lambda x: '.jpg' in x, image_paths)
    # Set the maximum dimension to which each image needs to be padded
    data = []
    [data.append(np.array(Image.open("images/" + path).convert('RGB'))) for path in image_paths]
    data = np.array(data)
    dimensions = []
    [dimensions.append(i.shape) for i in data]
    dimensions = pd.DataFrame(dimensions)
    x_max = max(dimensions.iloc[:,0])
    y_max = max(dimensions.iloc[:,1])
    return(x_max,y_max,data, image_paths)



class_names = []

with (open("classes.txt", "r")) as file: 
    class_names = file.read().strip().split("\n")

print(len(class_names))
images_list = os.listdir("images")
train_x_max, train_y_max, train_data, image_paths_train = get_maximum_dimensions(images_list)
train_images = create_image_array(train_x_max,train_y_max,train_data)
train_labels = [[7,6,8,9,17,19],[0,1,8,17]]
## Dimensions of the images
print(train_images.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2304, 4608,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

#image = Image.open("images/" + images_list[0])
# pad the images 

#print(np.array(image).shape)

#image = image.convert("RGB")
#print(np.array(image.convert("RGB")).shape)





