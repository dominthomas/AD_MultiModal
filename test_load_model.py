from tensorflow import keras

sagittal_model = keras.models.load_model('/home/dthomas/sagittal')

sagittal_model.summary()