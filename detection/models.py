
from django.db import models
import numpy as np
import keras,sys
import tensorflow.compat.v1 as tf
import os
from keras.models import load_model
from PIL import Image
import io,base64
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
import codecs
tf.experimental.output_all_intermediates(True)
graph=tf.get_default_graph()

class Photo(models.Model):
    image = models.ImageField(upload_to="photos")
    print(os.path.abspath('cnn_weights.h5'))
    IMAGE_SIZE=50
    MODEL_PATH= "./ml_models/cnn_weights.h5"
    MODEL_ARC="./ml_models/cnn_model.json"
    MODEL_MAST="./ml_models/model.h5"
    imagename = ['cat','bird','monkey']
    image_len = len(imagename)

    def predict(self):
        model=None
        global graph
        with graph.as_default():
            model=load_model(self.MODEL_MAST)
            #model = model_from_json(open(self.MODEL_ARC, 'r').read())
            #model.load_weights(self.MODEL_ARC)
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)

            image = Image.open(img_bin)
            image = image.convert("RGB")
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            data = np.asarray(image) / 255.0
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return self.imagename[predicted], percentage

    def image_src(self):
        with self.image.open() as img:
            base64_img=base64.b64encode(img.read()).decode()

            return "data:"+img.file.content_type+";base64,"+base64_img
