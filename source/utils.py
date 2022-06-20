import os
import cv2 # OpenCV for image editing, computer vision and deep learning
import base64 # Used for encoding image content string
import numpy as np # Numpy for math/array operations
from datetime import datetime, timedelta
from matplotlib import pyplot as plt # Matplotlib for visualization

def draw_rectangle(image, face):
    (start_x, start_y, end_x, end_y) = face["rect"]
    detection_rect_color_rgb = (0, 255, 255)
    cv2.rectangle(img = image, 
                  pt1 = (start_x - 3, start_y), 
                  pt2 = (end_x + 3, start_y - 50 ), 
                  color = detection_rect_color_rgb, 
                  thickness = -1)
    cv2.rectangle(img = image, 
                  pt1 = (start_x, start_y), 
                  pt2 = (end_x, end_y), 
                  color = detection_rect_color_rgb, 
                  thickness = 3)
    
    
    if face["prob"] != []:
        # Create probability text to be drawn over image
       # text = "{:.2f}%".format(face["prob"])
        text = face['prob']
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        probability_color_rgb = (0, 0, 0)
        cv2.putText(img = image, 
                    text = text, 
                    org = (start_x + 10, y), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1.2, 
                    color = probability_color_rgb, 
                    thickness = 2)
        
def draw_rectangles(image, faces):
    if len(faces) == 0:
        num_faces = 0
    else:
        num_faces = len(faces)
        for face in faces:
            draw_rectangle(image, face)
    return num_faces, image

def read_image(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("/home/site/wwwroot/data/" + str(datetime.now() + timedelta(hours=7))[:-7] + ".jpg", image)
    return image

def prepare_image(image):
    # Create string encoding of the image
    image_content = cv2.imencode('.jpg', image)[1].tostring()
    # Create base64 encoding of the string encoded image
    encoded_image = base64.b64encode(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return to_send

def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))