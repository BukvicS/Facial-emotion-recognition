import keras
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        width_shift_range   = 0.1           ,   #Pokusali smo sa 0.2 i dobijemo ispod 60% accuracy
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
            save_weights_only   = True          ,       #Cuvamo samo tezine, ne ceo model
            mode                = 'max'         ,
            verbose             = 1                     #Detaljnost prikaza
        ),

        EarlyStopping(
            monitor  = 'val_loss',              #Moze i val_accuracy, ali nije precizno jer moze da stagnira dok se val_loss smanjuje
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

def train_model( model, train_generator, validation_generator, callbacks, epochs = 50 ):
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

# ----------------------------- #

def plot_training_history( history ):
    epochs_count = len( history.history[ 'accuracy' ] )
    epochs_range = range( 1, epochs_count + 1 )

    figure, ( ax1, ax2 ) = plt.subplots( 1, 2, figsize = ( 14, 5 ) )

    ax1.plot( epochs_range,
              history.history[ 'accuracy' ],
              label      = 'Trening tacnost',
              color      = 'blue'           ,
              linewidth  = 2                ,
              marker     = 'o'              ,
              markersize = 3 
            )

    ax1.plot( epochs_range,
              history.history[ 'val_accuracy' ],
              label      = 'Validaciona tacnost',
              color      = 'red'                ,
              linewidth  = 2                    ,
              marker     = 's'                  ,
              markersize = 3 
            )
    
    ax1.fill_between( epochs_range,
                      history.history[ 'accuracy' ],
                      history.history[ 'val_accuracy' ],
                      color = 'lightgray',
                      alpha = 0.5,
                      label = 'Razlika izmedju treninga i validacije'
                    )
    
    # Označi najbolju validacionu tačnost
    best_val_acc = max( history.history[ 'val_accuracy' ] )
    best_epoch = history.history[ 'val_accuracy' ].index( best_val_acc ) + 1
    
    ax1.axvline( x = best_epoch,
                 color = 'green'    ,    
                 linestyle = '--'   ,
                 alpha = 0.7        , 
                 label = f'Najbolja epoha: { best_epoch }' 
                )
    ax1.scatter( best_epoch      ,
                 best_val_acc    ,
                 color  = 'green',
                 s      = 100    ,
                 zorder = 5
                )
    
    ax1.xaxis.set_major_locator( plt.MaxNLocator( nbins = 10, integer = True ) )
    ax1.set_xlim( [ 0.5, epochs_count + 0.5 ] )
    
    ax1.set_title( 'Tačnost modela kroz epohe', fontsize = 14, fontweight = 'bold' )
    ax1.set_xlabel( 'Epoha', fontsize = 12 )
    ax1.set_ylabel( 'Tačnost', fontsize = 12 )
    ax1.legend( loc = 'lower right', fontsize = 10 )
    ax1.grid( True, alpha = 0.3 )
    ax1.set_ylim( [0, 1.05] )

    ax2.plot( epochs_range,
              history.history[ 'loss' ], 
              label      = 'Trening gubitak' , 
              color      = 'blue'            , 
              linewidth  = 2                 , 
              marker     = 'o'               , 
              markersize = 3
            )
    
    ax2.plot( epochs_range,
              history.history[ 'val_loss' ], 
              label      = 'Validacioni gubitak' , 
              color      = 'red'                 , 
              linewidth  = 2                     , 
              marker     = 's'                   , 
              markersize = 3
            )
    
    # Označi najmanji validacioni gubitak
    best_val_loss = min( history.history[ 'val_loss' ] )
    best_loss_epoch = history.history[ 'val_loss' ].index( best_val_loss ) + 1
    
    ax2.xaxis.set_major_locator( plt.MaxNLocator( nbins = 10, integer = True ) )
    ax2.set_xlim( [ 0.5, epochs_count + 0.5 ] )
    
    ax2.axvline( x = best_loss_epoch, 
                 color = 'green'        , 
                 linestyle = '--'       , 
                 alpha = 0.7            ,
                 label = f'Najbolja epoha: { best_loss_epoch }'
                )
    ax2.scatter( best_loss_epoch,
                 best_val_loss      , 
                 color  = 'green'   , 
                 s      = 100       , 
                 zorder = 5
                )

    ax2.set_title( 'Gubitak modela kroz epohe', fontsize = 14, fontweight = 'bold' )
    ax2.set_xlabel( 'Epoha'  , fontsize = 12 )
    ax2.set_ylabel( 'Gubitak', fontsize = 12 )
    ax2.legend( loc='upper right', fontsize = 10 )
    ax2.grid( True, alpha = 0.3 )

    figure.suptitle( 'Prepoznavanje emocija na licu - Istorija treniranja', 
                     fontsize = 16      , 
                     fontweight = 'bold', 
                     y = 1.02
                    )
    
    plt.tight_layout()
    plt.savefig( 'training_history.png', 
                 dpi = 150, 
                 bbox_inches = 'tight'
                )
    
# ---------------------------- #

def plot_confusion_matrix( model                                                                                ,
                           test_generator                                                                       ,
                           weights_path = 'best_model.weights.h5'
                         ):
    
    model.load_weights( weights_path )

    #Predviđene i stvarne klase
    predictions = model.predict( test_generator )
    predicted_classes = tf.argmax( predictions, axis = 1 ).numpy()
    
    true_classes = test_generator.classes

    #Nazivi klasa
    class_names = list( test_generator.class_indices.keys() )

    cm = confusion_matrix( true_classes, predicted_classes )

    plt.figure( figsize = ( 10, 8 ) )
    
    sns.heatmap( cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                xticklabels = class_names,
                yticklabels = class_names
                )
    
    plt.title( 'Matrica konfuzije - Test skup', fontsize = 16, fontweight = 'bold' )
    plt.xlabel( 'Predviđene klase', fontsize = 12 )
    plt.ylabel( 'Stvarne klase'   , fontsize = 12 )
    plt.tight_layout()
    plt.savefig( 'confusion_matrix.png', dpi = 300, bbox_inches = 'tight' )

# ---------------------------- #

def main():
    train_generator, validation_generator = prepare_train_data( train_dir )
    test_generator = prepare_test_data( test_dir )

    model = create_cnn_model()
    model = compile_model( model, learning_rate = 0.001 )

    callbacks = set_callbacks( path = 'best_model.weights.h5' )

    model.summary()

    history = train_model( model, train_generator, validation_generator, callbacks, epochs = 50 )

    evaluate_model( model, test_generator, 'best_model.weights.h5' )

    plot_training_history( history )
    plot_confusion_matrix( model, test_generator, weights_path = 'best_model.weights.h5' )

# ----------------------------- #

if __name__ == '__main__':
    main()