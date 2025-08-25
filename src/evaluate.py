import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from data_loader import load_data
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

def evaluate_model(model_path='../models/best_model.h5'):
    _, test_gen, num_classes, class_names = load_data()
    model = load_model(model_path)
    
    # Predictions
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Metrics
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    auc = roc_auc_score(to_categorical(y_true, num_classes), y_pred, multi_class='ovr')
    print(f'ROC AUC: {auc:.4f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('../results/plots/confusion_matrix.png')
