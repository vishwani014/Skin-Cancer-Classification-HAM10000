# loading HAM10000 data (images + metadata CSV), preprocessing, augmentation, and splitting

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

DATA_DIR = ''
METADATA_PATH = '../data/HAM10000_metadata.csv'

def load_data(test_size=0.2, img_size=(224,224), batch_size=32):

    df = pd.read_csv(METADATA_PATH)
    df['image_path'] = df['image_id'].map(lambda x:os.path.join(DATA_DIR, f'{x}.jpg'))

    # Encode Labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])
    num_classes = len(le.classes_)

    # Train-test split (stratified for imbalance)
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)

    # Data augmentation for training (to handle imbalance and variety)
    # prevents overfitting to frequent classes.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # test generator (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generators
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='image_path', y_col='dx',
        target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col='image_path', y_col='dx',
        target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    return train_gen, test_gen, num_classes, le.classes_


# source venv/bin/activate

# touch .gitignore

# Add the following line to .gitignore:
# textdata/