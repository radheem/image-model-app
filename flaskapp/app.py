import os,io, base64
from flask import Flask, render_template, request
from matplotlib.figure import Figure
import matplotlib as plt
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './test_images/'
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024
model = tf.keras.models.load_model('../MLmodel/model_weights/simpleReluModel.h5')
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) #attach a sofmax to the final layer 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # set labels

def plot_image(img,predictions_array):
    fig = Figure(figsize=(20,10))
    ax = fig.subplots(1,2)
    ax[0].imshow(img)
    ax[1].set_xticklabels(class_names,rotation=40)
    ax[1].bar(class_names,predictions_array)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploading_file():
   if request.method == 'POST':
      f = request.files['file']
      up_fp = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
      f.save(up_fp)
      image = tf.keras.utils.load_img(
         up_fp,
         grayscale=True,
         color_mode='rgb',
         target_size=(28,28),
         interpolation='nearest',
         keep_aspect_ratio=True
      )
      input = (np.expand_dims(image,0))
      predictions_single = prob_model.predict(input)
      return plot_image(image,predictions_single[0])
		
if __name__ == '__main__':
   app.run(host="0.0.0.0",port=5000,debug = True)