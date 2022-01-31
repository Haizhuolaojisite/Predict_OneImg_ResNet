from torchvision import models
from PIL import Image
from torchvision import transforms
import torch

print(dir(models))

# Download the resnet-101 layers pre-trained model
resnet = models.resnet101(pretrained=True)

#Image needs to be preprocessed before passing into resnet model for prediction. TorchVision provides preprocessing class such as transforms for data preprocessing. transforms.preprocess method is used for preprocessing (converting the data into tensor). torch.unsqueeze is used for reshape, cropping, and normalizing the input tensor for feeding into network for evaluation
# Load the image

img_cat = Image.open("cat.jpeg").convert('RGB')
img_cat.show()



# Create a preprocessing pipeline
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# Pass the image for preprocessing and the image preprocessed
img_cat_preprocessed = preprocess(img_cat)

# Reshape, crop, and normalize the input tensor for feeding into network for evaluation
batch_img_cat_tensor = torch.unsqueeze(img_cat_preprocessed, 0)

# Resnet is required to be put in evaluation mode in order
# to do prediction / evaluation
resnet.eval()

# Get the predictions of image as scores related to how the loaded image
# matches with 1000 ImageNet classes. The variable, out is a vector of 1000 scores
out = resnet(batch_img_cat_tensor)

# Load the file containing the 1,000 labels for the ImageNet dataset classes
#
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
# Find the index (tensor) corresponding to the maximum score in the out tensor.
# Torch.max function can be used to find the information
_, index = torch.max(out, 1)
# Find the score in terms of percentage by using torch.nn.functional.softmax function
# which normalizes the output to range [0,1] and multiplying by 100
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# Print the name along with score of the object identified by the model
print(labels[index[0]], percentage[index[0]].item())
# Print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores.
_, indices = torch.sort(out, descending=True)
top5 = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print('Top 5 Classes',top5)
