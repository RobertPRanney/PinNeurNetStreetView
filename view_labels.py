import os
from PIL import Image
import random
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from time import sleep

num = 20




if __name__ == '__main__':
    test_datagen = ImageDataGenerator(rescale=1/255)
    model = load_model('first_model_out.h5')

    print("Should be zeros")
    for i in range(num):
        sleep(0.5)
        path = random.choice(os.listdir('data_small/validation/0'))
        path = 'data_small/validation/0/'+path
        img = image.load_img(path, target_size=(640,640))
        img.show()

        img = image.load_img(path, target_size=(20,20))


        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        print(model.predict(img)[0][0])

    print("\n\nShould be ones")
    for i in range(num):
        sleep(0.5)
        path = random.choice(os.listdir('data_small/validation/1'))
        path = 'data_small/validation/1/'+path
        img = image.load_img(path, target_size=(640,640))
        img.show()

        img = image.load_img(path, target_size=(20,20))


        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        print(model.predict(img)[0][0])
