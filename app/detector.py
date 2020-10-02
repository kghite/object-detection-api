"""
Object Detection API

Serve endpoints for server status and object detection related to hop navigation
See README for updated endpoint list
"""

import os
import io
from flask import Flask, request, jsonify, render_template, abort

from PIL import Image
import numpy as np
import cv2

# Internal
import model_interface as detector


app = Flask(__name__, static_url_path='/static')


"""
Serve app endpoint list on home page
"""
@app.route('/')
def serve_index():
	return render_template('index.html')


"""
Return app status - used as wake ping for heroku server on robot startup
"""
@app.route('/api/v1/status')
def ping_server():
	status_code = flask.Response(status=200)

	return status_code


"""
Detect objects in posted image
"""
@app.route('/api/v1/detect', methods=['POST'])
def detect_objects():
	# Verify image
	file = request.files['image']
	image_extensions=['jpg', 'jpeg', 'png']
	if file.filename.split('.')[1] not in image_extensions:
		abort(404, description="Incorrect image format")
	
	# Load vars
	image_bytes = file.read()
	pil_image = Image.open(io.BytesIO(image_bytes))
	nparr = np.frombuffer(image_bytes, np.uint8)
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	
	if (detector.detect_dog(pil_image)):
		dog_breed = detector.predict_breed_transfer(pil_image)
		return jsonify ('This dog is a {}'.format(dog_breed))
	elif (detector.detect_face(img_np)):
		dog_breed = detector.predict_breed_transfer(pil_image)
		return jsonify('This human resembles a {}'.format(dog_breed))
	else:
		return jsonify('No human face or dog detected')


if __name__ == '__main__':
	app.run(debug=True, port=os.getenv('PORT',5000))