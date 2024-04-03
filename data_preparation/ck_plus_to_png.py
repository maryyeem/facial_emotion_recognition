import numpy as np
import matplotlib.pyplot as plt

X = np.load('npy_files/X.npy')
y8 = np.load('npy_files/y.npy')
y = reindex_labels(y8)

def reindex_labels(y8):
    y = np.zeros(y8.shape, dtype=np.int8)
    label_mapping = {0:6, 2:-1, 1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
    for i in range(0,len(y8)):
        y[i] = label_mapping[y8[i]]

    return y


emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

for i in range(0,10):
    plt.xlabel(emotions[y[i]])
    plt.imshow(X[i].reshape(96,96),cmap='gray')
    plt.show()
    
# # from os import mkdir
# # for emotion in emotions:
# #     mkdir(f'C:/Users/Maryem/Desktop/ck+/ck-images' + f'{emotion} ' + f'{emotions[emotion]}')

# from PIL import Image

# count = 0
# for i in range(0,X.shape[0]):
#     count_string = str(count).zfill(7)
#     fname = 'C:/Users/Maryem/Desktop/ck+/ck-images' + f'{y[i]} ' + f'{emotions[y[i]]}/' + f'{emotions[y[i]]}-{count_string}.png'
#     image_array = X[i].astype(np.uint8)
# # Convert the NumPy array to an image object
#     image = Image.fromarray(image_array.reshape((96,96)))
#     image.save(fname) 
#     count += 1