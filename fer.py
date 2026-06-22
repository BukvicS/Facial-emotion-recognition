import keras
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dir = 'input/train'
test_dir  = 'input/test'

datagen = ImageDataGenerator(
    rescale=1./255,             # normalizacija piksela na opseg [0,1]
    validation_split=0.2        # 20% podataka (slika) ide na validaciju
)

# 80% podataka za trening
train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),       #Dimenzije slika
    batch_size=128,             #Broj SLIKA obradenih odjednom
    color_mode='grayscale',     #48x48x1 velicina jedne slike (RGB -> 48x48x3)
    class_mode='categorical',   #7 kategorija
    subset='training',          #Povlaci 80% podataka (20 smo odbili za validaciju)
    shuffle=True                #Catastrofic forgetting ukoliko nema mesanja podataka 
)

# 20% podataka za validaciju
validation_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    batch_size=128,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# CNN
model = Sequential([
    keras.Input(shape=(48, 48, 1)),
    
    Conv2D(32, kernel_size=(3, 3), activation='relu'),  #32 karakteristicne tacke po kojima trazi slicnosti i razlicitosti u kategorijama
    MaxPooling2D(pool_size=(2, 2)),                     #Smanjivanje rezolucije, ostavljanje samo najjacih piksela u kvadratima 2x2
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  #Svaki sledeci sloj istrazuje oko karakteristicnih tacki iz prethodnog sloja, i trazi jos detaljnije slicnosti i razlicitosti
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),      #2D -> 1D matricu
    
    # Potpuno povezani slojevi
    Dense(128, activation='relu'),  # Izvlacenje zakonitosti po kojima slicne linije i oblici pripadaju istoj kategoriji
    Dense(7, activation='softmax')  # 7 klasa (emocije)
])

# Kompajliranje modela
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Callback za čuvanje najboljih težina
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

# Treniranje modela
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint_callback],
    epochs=50
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(48, 48),
    batch_size=128,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

model.load_weights('best_model.weights.h5')

test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Gubici  na test skupu: {test_loss:.4f}')
print(f'Tacnost na test skupu: {test_accuracy:.4f}')
