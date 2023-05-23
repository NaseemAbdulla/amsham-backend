import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import base64
import io
import numpy as np

def load_model():
    model = tf.keras.models.load_model("model/model.h5")
    return model  


def pre_process(image_file):
    
	# padding if needed
	# pad = len(image_file)%4
	# pad += len(image_file)
	# image = image_file.ljust(pad, "=")
	# Decode base64-encoded imag data
	image = base64.b64decode(image_file)
 

	# get the image file name for saving output later on
	image = cv2.imdecode(np.frombuffer(image,
										np.uint8),
						cv2.IMREAD_COLOR)
	image = cv2.resize(image, (224,224))
	image = np.expand_dims(image,axis=0)
	image = image/255.0

	return image



def post_process(prediction):
	print(prediction)
	classes = (
    	0, 1, 2
	)
	symbol = (
		"&#3443;", "&#3444;", "&#3419;"
	)
	fraction = (
		"1/4", "1/2", "1/20"
	)
	output = np.argmax(prediction)
	print(f"output = {output} predection = {prediction}")
	res = {"class": str(output),"symbol": None, "fraction": None }
	for c in classes:
		if(str(output) == str(c)):
			res["symbol"] = symbol[c]
			res["fraction"] = fraction[c]
			
	return res