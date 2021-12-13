
from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
from keras_preprocessing import image


target_size = (256,256)
path = 'cat1.jpg'


show_img = image.load_img(path)
test_image = image.load_img(path, target_size=target_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

cls = keras.models.load_model("classifier")

result = result = cls.predict(test_image)
wynik = ""
if result[0][0]:
    wynik = "!!!TO JEST KOT EGZOTYCZNY!!!"
elif result[0][1]:
    wynik = "!!!TO JEST KOT MAINE COON!!!"   
elif result[0][2]:
    wynik = "!!!TO JEST KOT ROSYJSKI NIEBIESKI!!!"
    
plt.title(wynik)
plt.imshow(show_img,aspect='equal')
plt.show()
print(result[0])