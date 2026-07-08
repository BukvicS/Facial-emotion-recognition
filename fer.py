import keras
import tensorflow as tf 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ----------------------------- #

train_dir = 'input/train'
test_dir  = 'input/test'

# ----------------------------- #
 
def prepare_train_data( train_dir ):

    train_datagen = ImageDataGenerator(
        rescale             = 1./255        ,
        validation_split    = 0.2           ,
        zoom_range          = 0.2           ,
        width_shift_range   = 0.1           ,
        height_shift_range  = 0.1           ,
        horizontal_flip     = True          ,
        rotation_range      = 20            ,
        brightness_range    = [ 0.8, 1.2 ]  ,
        shear_range         = 0.15          ,
        fill_mode           = 'nearest'
    )   

    validation_datagen = ImageDataGenerator(
        rescale = 1./255,
        validation_split = 0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory   = train_dir     ,
        target_size = ( 48, 48 )    ,
        batch_size  = 128           ,
        color_mode  = 'grayscale'   ,
        class_mode  = 'categorical' ,
        subset      = 'training'    ,
        shuffle     = True
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory   = train_dir     ,
        target_size = ( 48, 48 )    ,
        batch_size  = 128           ,
        color_mode  = 'grayscale'   ,
        class_mode  = 'categorical' ,
        subset      = 'validation'  ,
        shuffle     = True
    )
    
    return train_generator, validation_generator

# ----------------------------- #

def prepare_test_data( test_dir ):

    test_datagen = ImageDataGenerator(
        rescale = 1./255
    )

    test_generator = test_datagen.flow_from_directory(
        directory   = test_dir      ,
        target_size = (48, 48)      ,
        batch_size  = 128           ,
        color_mode  = 'grayscale'   ,
        class_mode  = 'categorical' ,
        shuffle     = False
    )
    
    return test_generator

# ----------------------------- #

def create_cnn_model( input_shape = ( 48, 48, 1 ), num_classes = 7 ):

    model = Sequential([
        keras.Input( shape = (48, 48, 1) ),
        
        Conv2D( 32, kernel_size = (3, 3), activation='relu', padding='same' ),  #32 karakteristicne tacke po kojima trazi slicnosti i razlicitosti u kategorijama
        BatchNormalization(),
        
        Conv2D( 64, kernel_size = (3, 3), activation='relu', padding='same' ),  #Svaki sledeci sloj istrazuje oko karakteristicnih tacki iz prethodnog sloja, i trazi jos detaljnije slicnosti i razlicitosti
        BatchNormalization(),
        MaxPooling2D( pool_size = (2, 2) ),
        Dropout( 0.25 ),
        
        Conv2D( 128, kernel_size = (3, 3), activation='relu', padding='same' ),
        BatchNormalization(),
        MaxPooling2D( pool_size = (2, 2) ),
        Dropout( 0.25 ),
 
        Conv2D( 256, kernel_size = (3, 3), activation='relu', padding='same' ),
        BatchNormalization(),
        Dropout( 0.25 ),

        Conv2D( 256, kernel_size = (3, 3), activation='relu', padding='same' ),
        BatchNormalization(),
        MaxPooling2D( pool_size = (2, 2) ),
        Dropout( 0.25 ),

        Flatten(),      #2D -> 1D matricu
        
        Dense( 256, activation='relu' ),  # Izvlacenje zakonitosti po kojima slicne linije i oblici pripadaju istoj kategoriji
        BatchNormalization(),
        Dropout( 0.5 ),

        Dense( 7, activation='softmax' )  # 7 klasa (emocije)
    ])

    return model

# ----------------------------- #

def compile_model( model, learning_rate = 0.001 ):
    model.compile(
        loss        = 'categorical_crossentropy',
        optimizer   = tf.keras.optimizers.Adam( learning_rate = learning_rate ),
        metrics     = ['accuracy']
    )
    return model

# ----------------------------- #

def set_callbacks( path = 'best_model.weights.h5' ):
    callbacks = [
        ModelCheckpoint(
            filepath            = path          ,
            monitor             = 'val_accuracy',
            save_best_only      = True          ,
            save_weights_only   = True          ,
            mode                = 'max'         ,
            verbose             = 1     #Detaljnost prikaza
        ),

        EarlyStopping(
            monitor  = 'val_loss',     #Moze i val_accuracy, ali nije precizno jer moze da stagnira dok se val_loss smanjuje
            patience = 10        ,
            verbose  = 1         ,              
            restore_best_weights = True
        ),

        ReduceLROnPlateau(
            monitor  = 'val_loss'   ,
            factor   = 0.3          ,   #Kad nema poboljsanja, smanjuje se LR mnozenjem sa faktorom
            patience = 5            ,
            verbose  = 1            ,
            min_lr = 1e-6               #Proba, eksperimentalno 
        )
        
    ]
    return callbacks

# ----------------------------- #

def train_model( model, train_generator, validation_generator, callbacks, epochs = 70 ):
    history = model.fit(
        train_generator                                     ,
        steps_per_epoch     = len( train_generator )        ,
        validation_data     = validation_generator          ,
        validation_steps    = len( validation_generator )   ,
        callbacks           = callbacks                     ,
        epochs              = epochs
    )

    return history

# ----------------------------- #

def evaluate_model( model, test_generator, weights_path ):

    model.load_weights( weights_path )

    test_loss, test_accuracy = model.evaluate( test_generator )

    print(f'Gubici  na test skupu: {test_loss:.4f}')
    print(f'Tacnost na test skupu: {test_accuracy:.4f}')

    return test_loss, test_accuracy

# ----------------------------- #

def main():
    train_generator, validation_generator = prepare_train_data( train_dir )
    test_generator = prepare_test_data( test_dir )

    model = create_cnn_model()
    model = compile_model( model, learning_rate = 0.001 )

    callbacks = set_callbacks( path = 'best_model.weights.h5' )

    model.summary()

    history = train_model( model, train_generator, validation_generator, callbacks, epochs = 50 )

    test_loss, test_accuracy = evaluate_model( model, test_generator )

# ----------------------------- #

if __name__ == '__main__':
    main()