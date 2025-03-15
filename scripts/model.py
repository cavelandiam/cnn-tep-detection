from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, TimeDistributed, Flatten, Conv3D, MaxPooling3D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from utils.config import RADIMAGENET_WEIGHTS

def build_efficientnet_feature_extractor():
    """Carga EfficientNetB0 preentrenado en RadImageNet."""
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.load_weights(RADIMAGENET_WEIGHTS)
    
    for layer in base_model.layers:
        layer.trainable = False  # Congelamos capas preentrenadas

    return Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

def build_3d_cnn():
    """Construye una 3D-CNN que recibe los features extraídos por EfficientNet."""
    input_shape = (64, 224, 224, 3)
    inputs = Input(shape=input_shape)

    efficientnet = build_efficientnet_feature_extractor()
    feature_extractor = TimeDistributed(efficientnet)(inputs)

    x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu")(feature_extractor)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
