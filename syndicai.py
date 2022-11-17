import numpy as np 
import cv2
from keras import models
from utils import url_to_image, b64_to_image, image_to_base64


#procesamos las imagenes
from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)
test_data= ImageDataGenerator(rescale = 1. / 255)


training_set = train_data.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode = 'binary'
)
validation_set = val_data.flow_from_directory(
    validation_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)
test_set = test_data.flow_from_directory(
    test_dir,
    target_size=(64,64),
    class_mode='binary'     
)

#Hacemos la imagen en 64,64,3

class PythonPredictor:

    def __init__(self, config):
      red = models.load_model("/RedCNN_PerrosyGatos.h5")
        
    def predict(self, payload):

        # Obtener la imagen por el URL  
        try: 
            image = url_to_image(payload["image_url"])
            res= red.predict(image)
            return np.round(res)
        except:
            image = b64_to_image(payload["image_b64"])       
                       
        return image_to_base64(image)