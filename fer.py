import keras
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_dir = 'input/train'
test_dir  = 'input/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,          #Pokusali smo sa 0.2 i dobijemo ispod 60% accuracy
    height_shift_range=0.1,
    rotation_range=20,
    fill_mode='nearest',
)   

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

# 80% podataka za trening
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),       #Dimenzije slika
    batch_size=128,             #Broj SLIKA obradenih odjednom
    color_mode='grayscale',     #48x48x1 velicina jedne slike (RGB -> 48x48x3)
    class_mode='categorical',   #7 kategorija
    subset='training',          #Povlaci 80% podataka (20 smo odbili za validaciju)
    shuffle=True                #Catastrofic forgetting ukoliko nema mesanja podataka 
)

# 20% podataka za validaciju
validation_generator = validation_datagen.flow_from_directory(
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
    BatchNormalization(),                               #Normalizacija podataka
    Dropout(0.25),                                      #Random iskljucivanje 25% neurona, jure se prave karakteristike, protiv overfittinga
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  #Svaki sledeci sloj istrazuje oko karakteristicnih tacki iz prethodnog sloja, i trazi jos detaljnije slicnosti i razlicitosti
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),      #2D -> 1D matricu
    
    # Potpuno povezani slojevi
    Dense(128, activation='relu'),  # Izvlacenje zakonitosti po kojima slicne linije i oblici pripadaju istoj kategoriji
    BatchNormalization(),
    Dropout(0.5),

    Dense(7, activation='softmax')  # 7 klasa (emocije)
])

# Kompajliranje modela
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Callback-ovi za cuvanje najboljeg modela 
callbacks = [
    ModelCheckpoint(
        filepath='best_model.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1               #Detaljnost prikaza
    ),

    EarlyStopping(
        monitor='val_loss',     #Moze i val_accuracy, ali nije precizno jer moze da stagnira dok se val_loss smanjuje
        patience=10,
        verbose=1,              
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,         #Kad nema poboljsanja, smanjuje se LR mnozenjem sa faktorom
        patience=5,
        verbose=1,
        min_lr=1e-6         #Proba, eksperimentalno 
    )
    
    ]

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=callbacks,
    epochs=50
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

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

