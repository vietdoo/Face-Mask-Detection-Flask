import cv2
import numpy as np 
from source.utils import get_folder_dir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model

def detect_faces_with_ssd(image, min_confidence = 0.2):
    faces_list = []
    models_dir = get_folder_dir("models") 
    prototxt_filename = "deploy.prototxt.txt"
    model_filename = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(models_dir + prototxt_filename, 
                                   models_dir + model_filename)
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
            face_dict['prob'] = confidence * 100
            faces_list.append(face_dict)
    return faces_list

def run_model(img):
    faces = detect_faces_with_ssd(img)
    new_img = img
    model5 = load_model('P:/Mask/models/masknet.h5')
    mask_label = {0:'MASK',1:'KHONG MASK'}
    dist_label = {0:(0,255,0),1:(255,0,0)}
    if len(faces) >= 1:
        label = [0 for i in range(len(faces))]
        #colored output image
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]["rect"]
            crop = new_img[y:h,x:w]
            crop = cv2.resize(crop,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            mask_result = model5.predict(crop)
            faces[i]['prob'] = mask_label[mask_result.argmax()]
           # cv2.putText(new_img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
    return faces