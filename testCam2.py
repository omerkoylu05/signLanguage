import cv2
from docutils.nodes import classifier
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import ast

# Open the text file and read the class names
# with open('class_names.txt', 'r') as f:
#     class_names_str = f.read()
#
# # Use literal_eval to convert the string back into a list
# class_names = ast.literal_eval(class_names_str)
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','hastane','i', 'j', 'k', 'l', 'm','manifest', 'n', 'o', 'p','polis', 'r', 's','saat', 't', 'u', 'v', 'y', 'z', 'ç', 'ö', 'ü', 'ğ', 'ı', 'ş']  # Toplamda 33 sınıf olduğunu belirtmiştiniz.
# Specify the directory you want to list
# dir_path = './dataset'

# Get a list of all the directory names in the specified directory
# class_names = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

print(class_names)
# Load the trained model
model = load_model('./model/model_3_0.95.keras')

from keras.utils import plot_model
plot_model(model, to_file='model.png')

# Open the camera
cap = cv2.VideoCapture("/dev/video1")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Check if the camera opened successfully
while cap.isOpened():
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    # Capture a single frame
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    # Resmi BGR'den RGB'ye dönüştür
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resmi HSV renk uzayına dönüştür
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # cv2.imshow("hsv",hsv_image)
    # Genişletilmiş cilt rengi aralığını belirle
    lower_skin = np.array([0, 24, 40], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Maskeleme
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    # cv2.imshow("cilt",skin)

    background_color = (100, 100, 125)  # Beyaz renk
    background = np.full_like(image_rgb, background_color)

    # Cilt rengi maskesi dışındaki bölgeyi arka planla doldur
    non_skin_mask = cv2.bitwise_not(skin_mask)
    # print("non_skin_mask",non_skin_mask.shape)
    # print("background",background.shape)
    background=cv2.bitwise_and(background,background,mask=non_skin_mask)
    result = cv2.bitwise_or(skin, background)

    # Geri kalan kısmı griye dönüştür
    # gray_mask = cv2.bitwise_not(skin_mask)
    # gray_background = cv2.cvtColor(cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2GRAY)
    #
    # # Gri arka planı oluştur
    # gray_background_rgb = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2RGB)

    # Cilt rengi alanlarını renkli arka plan ile birleştir
    # result = cv2.bitwise_or(skin, background)

    (h, w) = frame.shape[:2]

    # Görüntünün merkezini belirle
    center = (w / 2, h / 2)

    # 90 derece döndürme matrisini oluştur
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Döndürme matrisini ayarla
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Döndürme matrisini uygula
    # frame = cv2.warpAffine(frame, M, (nW, nH))

    frame=result[0:640,80:560]
    # frame = cv2.resize(frame, (400, 800))

    # Check if the frame has been captured successfully
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    # Preprocess the image
    img = cv2.resize(frame, (320, 240))  # Resize image to match the input size expected by the model
    # img_input = img.astype('float32') / 255  # Normalize pixel values to [0, 1]
    # print(img_input.shape)
    img_input=image.img_to_array(img)
    img_input/=255.
    img_input = np.expand_dims(img_input, axis=0)  # Add an extra dimension for batch size

    # Use the model to predict the class of the image
    predictions = model.predict(img_input, verbose=0)
    # print(predictions)

    # Get the index of the class with the highest predicted probability
    predicted_class = np.argmax(predictions)
    percent = predictions[0][predicted_class]
    class_name = "???"

    if percent > 0.30:
        print(predicted_class)
        print(percent)
        class_name = class_names[predicted_class]
        print('The predicted class name of the image is:', class_name)

    cv2.putText(img, 'Yuzde:%'+str(predictions[0][predicted_class]*100), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, 'Harf:' + class_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
    # print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    cv2.imshow('frame', img)

    # Release the camera
    if cv2.waitKey(5) == 27:
        break
    pass

cv2.destroyAllWindows()
cap.release()

