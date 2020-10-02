"""
Provide object detection model handler class
"""

import torch
from torch import nn
from torchvision import models, transforms


use_cuda = torch.cuda.is_available()

# Load pretrained model
model = models.densenet121(pretrained=False)
model.classifier=nn.Sequential(nn.Linear(1024,512), nn.ReLU(),
											nn.Dropout(0.2),
											nn.Linear(512,133))
model.load_state_dict(torch.load('model_transfer.pt',map_location='cpu'))


"""
Predict ImageNet class with pretrained VGG-16 model

Input: img_path -> path to an image
	
Returns: class prediction index
"""
def resnet50_predict(pil_image):
	resnet50 = models.resnet50(pretrained=True)
	mean=[0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image_transforms = transforms.Compose([transforms.Resize(256),
											transforms.CenterCrop(224),
											transforms.ToTensor(),
											transforms.Normalize(mean,std)])
	image_tensor = image_transforms(pil_image)
	image_tensor.unsqueeze_(0)
	resnet50.eval()

	if use_cuda:
		image_tensor = image_tensor.cuda()

	output = resnet50(image_tensor)
	_,classes = torch.max(output,dim=1)

	return classes.item() # predicted class index


"""
Detect human faces

Returns: bool -> face present
"""
def detect_face(pil_image):
	faces = face_recognition.face_locations(pil_image)
	return len(faces) > 0


"""
Detect dogs

Returns: bool -> dog present
"""
def detect_dog(pil_image):
	class_dog=resnet50_predict(pil_image)

	return class_dog >= 151 and class_dog <=268


"""
Classify dog breed
	
Returns: dog breed class index
"""
def predict_breed_transfer(pil_image):
	# Load image
	mean_train_set,std_train_set = [0.487,0.467,0.397],[0.235,0.23,0.23]
	image_transforms= transforms.Compose([transforms.Resize(256),
											transforms.CenterCrop(224),
											transforms.ToTensor(),
											transforms.Normalize(mean_train_set,std_train_set)])
	image_tensor = image_transforms(pil_image);
	image_tensor.unsqueeze_(0)

	if use_cuda:
		image_tensor = image_tensor.cuda()

	# Run classifier
	model.eval()
	output = model(image_tensor)
	_,class_idx=torch.max(output,dim=1)
	f = open('dog_breeds.txt')
	class_names = f.readlines()
	class_names = [class_name[:-1] for class_name in class_names]

	return class_names[class_idx]