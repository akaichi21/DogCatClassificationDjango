from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
import cv2
import numpy as np

model = load_model('./model/dog_cat_classification.h5')

# Create your views here.
def home(request):
    return render(request, 'app/pages/home.html')

def services(request):
    context = {}
    context['service'] = ''
    context['result'] = ''
    IMAGE_HEIGHT = 100
    IMAGE_WIDTH = 100
    LABEL_DOG = np.array([[1.]], dtype='float32')
    LABEL_CAT = np.array([[0.]], dtype='float32')
    result = ""
    if request.method == 'POST':
        if request.FILES.get('img') != None:
            uploaded_file = request.FILES['img']
            context['service'] = 'img'
        fileSystemStorage = FileSystemStorage()
        fileSystemStorage.save(uploaded_file.name, uploaded_file)
        context['url'] = fileSystemStorage.url(uploaded_file)
        img = '.' + context['url']
        image = cv2.imread(img)
        image_resize = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        image_test = image_resize.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        predict = model.predict(image_test)
        if (predict == LABEL_DOG):
            result += "chó"
        elif (predict == LABEL_CAT):
            result += "mèo"
        context['result'] = result

    return render(request, 'app/pages/services.html', context)

def about(request):
    return render(request, 'app/pages/about.html')

def contact(request):
    return render(request, 'app/pages/contact.html')
