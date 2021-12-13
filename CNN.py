from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from matplotlib import pyplot
import sys

from keras_preprocessing.image import ImageDataGenerator

# wartości sterowane
epochs = 50
batch = 4
target_size = (256, 256)


# przygotowanie bazy(normalizacja wartości oraz augmentacja)
train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = target_size, class_mode='categorical', batch_size=batch)

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = target_size, class_mode='categorical', batch_size=batch)


x, y = target_size
# stworzenie modelu sieci
classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(x, y, 3)))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dense(3, activation='softmax'))

classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])


# rysowanie wykresów
def summarize_diagnostics(history):
	# narysuj wykres strat
	pyplot.subplot(211)
	pyplot.ylim((0,1.5))
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	pyplot.legend()
	pyplot.xlabel("Epochs")
	# narysuj wykres precyzji 
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.ylim((0.5,1))
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	pyplot.legend()
	pyplot.xlabel("Epochs")
	
	# zapisz wykres do pliku
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


# wytrenuj sieć
history = classifier.fit(training_set, steps_per_epoch=len(training_set),
	validation_data=test_set, validation_steps=len(test_set), epochs = epochs, verbose=1)
# sprawdź sieć
_, acc = classifier.evaluate(test_set, steps=len(test_set), verbose=0)

summarize_diagnostics(history)


print('> %.3f' % (acc * 100.0))
# learning curves

classifier.save('classifier')
