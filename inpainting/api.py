from flask import Flask, jsonify, request, after_this_request
import os
import uuid
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import ssl

from werkzeug.exceptions import HTTPException
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

import tensorflow as tf
from model import Image_inpainting



executor = ThreadPoolExecutor(1)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'inpainting.sqlite')
db = SQLAlchemy(app)
ma = Marshmallow(app)

Image_inpainting = Image_inpainting()

class ImgDB(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    imguid = db.Column(db.String(120), unique=True)
    imgbase64 = db.Column(db.String(120), unique=False)

    def __init__(self, imguid, imgbase64):
        self.imguid = imguid
        self.imgbase64 = imgbase64

class ImgSchema(ma.Schema):
    class Meta:
        # Fields to expose
        fields = ('imguid', 'imgbase64')


img_schema = ImgSchema()
imgs_schema = ImgSchema(many=True)

def task_error(status_code, message, UUID):
    response = jsonify({
        'status': status_code,
        'message': message,
        "task_uid": UUID,
    })
    #response.status_code = status_code
    return response

def task_success(task_url, UUID):
    response = jsonify({
        'task_url': task_url,
        "task_uid": UUID,
    })
    return response

@app.route('/inpainting/api/v1.0/test')
def hello_world():
    return 'Hello, World!'

@app.route('/inpainting/api/v1.0/order', methods = ['POST'])
def postJsonHandler_multiprocessing():
   
    json_content = request.get_json(force=True)
    imguid = str(uuid.uuid1())
    
    if Image_inpainting.json_image_checker(json_content['image']) !=1:
        return task_error(1001, 'Invalid URL of Image', imguid)   
    if Image_inpainting.json_mask_checker(json_content['image_mask'])!=1:
        return task_error(1002, 'Invalid URL of Mask', imguid)
    
    if executor.submit(InpaintingHandler, imguid)!=1:
        return task_success(json_content['image'],imguid)
    else:     
        return task_error(1003, 'Internal Server Error', imguid)
    
def InpaintingHandler(imguid):
    import base64
    os.system('python inpainting_run.py')
    dict_item = base64.b64encode(open('output.png', 'rb').read()).decode('utf-8')
    new_img_post = ImgDB(imguid, dict_item)
    db.session.add(new_img_post)
    db.session.commit()
    print("Task finished - imguid: %s" % (imguid))
    
    return True

# endpoint to get base64 by uuid
@app.route("/inpainting/api/v1.0/order/<inputuid>", methods=["GET"])
def img_detail(inputuid):
    uuid = ImgDB.query.filter_by(imguid=inputuid).first()
    return jsonify(uuid.imgbase64)

##Supervisior function##

# endpoint to show all imgs
@app.route("/inpainting/api/v1.0/order/storage", methods=["GET"])
def get_user():
    all_imgs = ImgDB.query.all()
    result = imgs_schema.dump(all_imgs)
    return jsonify(result)

# endpoint to delete base64
@app.route("/inpainting/api/v1.0/order/storage/<inputuid>", methods=["DELETE"])
def img_delete(inputuid):
    uuid = ImgDB.query.filter_by(imguid=inputuid).first()
    id = ImgDB.query.get(uuid.id)
    db.session.delete(id)
    db.session.commit()

    return img_schema.jsonify(id)

if __name__ == '__main__':
    app.run(host='172.19.140.150', port=5003)
    
    

    