import os
from flask import Flask,jsonify,request,render_template,send_from_directory
from source.face_detection import run_model
from source.utils import draw_rectangles, read_image, prepare_image
from config import DETECTION_THRESHOLD

app = Flask(__name__, static_folder='static',)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/love')
def love():
  return "HELLO"

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    # Read image
    image = read_image(file)
    # Detect faces
    faces = run_model(image)
    
    # Draw detection rects
    num_faces, image = draw_rectangles(image, faces)
    to_send = prepare_image(image)
    print(faces)
    return render_template('index.html', face_detected=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)

if __name__ == '__main__':
    app.run(debug = True, use_reloader = True)
