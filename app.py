import os
from flask import Flask,jsonify,request,render_template,send_from_directory
from source.utils import draw_rectangles, read_image, prepare_image
from config import DETECTION_THRESHOLD
import cv2
import numpy as np 
import os
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
app = Flask(__name__, static_folder='static',)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model5 = load_model(os.getcwd() +'/models/masknet.h5')

def detect_faces_with_ssd(image, min_confidence = 0.39):
    faces_list = []
    prototxt_filename = "/models/deploy.prototxt.txt"
    model_filename = "/models/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(os.getcwd() + prototxt_filename, 
                                   os.getcwd() + model_filename)
    (image_height, image_width) = image.shape[:2]
    resized_image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image,
                                 scalefactor = 1.0, 
                                 size = (300, 300), 
                                 mean = (104.0, 177.0, 123.0))
    net.setInput(blob)
    detected_faces = net.forward()
    num_detected_faces = detected_faces.shape[2]
    for index in range(0, num_detected_faces):
        face_dict = {}
        confidence = detected_faces[0, 0, index, 2]
        confidence = confidence.item()
        if confidence > min_confidence:
            rect = detected_faces[0, 0, index, 3:7] * np.array([image_width, image_height, image_width, image_height])
            (start_x, start_y, end_x, end_y) = rect.astype("int")
            start_x = start_x.item()
            start_y = start_y.item()
            end_x = end_x.item()
            end_y = end_y.item()
            start_x = max(0,start_x)
            start_y = max(0,start_y)
            end_x = min(end_x,image_width)
            end_y = min(end_y,image_height)
            face_dict['rect'] = (start_x, start_y, end_x, end_y)
            face_dict['prob'] = 'ok'
            faces_list.append(face_dict)
    return faces_list

def run_model(img):
    faces = detect_faces_with_ssd(img)
    
    mask_label = {0:'MASK',1:'NO MASK'}
    dist_label = {0:(0,255,0),1:(0,0,255)}
    if len(faces) >= 1:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]["rect"]
            yy = h - y
            xx = w - x
            if (xx > yy):
                y = y - (xx - yy) // 2
                h = y + xx
            if (xx < yy):
                x = x - (yy - xx) // 2
                w = x + yy
            crop = img[y:h,x:w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            mask_result = model5.predict(crop)
            faces[i]['prob'] = mask_label[mask_result.argmax()]
            faces[i]['color'] = dist_label[mask_result.argmax()]
    return faces

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    image = read_image(file)
    faces = run_model(image)
    num_faces, image = draw_rectangles(image, faces)
    to_send = prepare_image(image)
    print(faces)
    return render_template('index.html', face_detected=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)

if __name__ == '__main__':
    
    app.run(debug = False)
