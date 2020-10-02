import requests
from PIL import Image
from io import BytesIO
import os

def upload(image_stream):
	api_url = 'http://localhost:5000/api/v1/detect'
	r = requests.post(api_url, files={'image': 
										('test.jpg', image_stream, 'image/png')})

	return r.content

if __name__ == '__main__':
	image = Image.open("test.jpg")
	byte_io = BytesIO()
	image.save(byte_io, 'png')
	byte_io.seek(0)
	print(upload(byte_io))