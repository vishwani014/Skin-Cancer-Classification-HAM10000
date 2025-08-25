import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Adam
from data_loader import load_data
from model import build_model
import numpy as np
import json

def train_model(epochs=50, batch_size=32):
    train_gen, test_gen, num_classes, class_names = load_data(batch_size=batch_size)

    # Compute class weights for imbalance
    y_train = train_gen.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = build_model(num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('../models/best_model.h5', monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    metrics = {
        'train_accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1]
    }
    with open('../results/metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Unfreeze some layers for fine-tuning
    for layer in model.layers[-20:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=20, validation_data=test_gen, callbacks=callbacks)
    
    return model, history