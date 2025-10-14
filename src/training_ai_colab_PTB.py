# ===================================================================
# CNN 1D TRAINING WITH DATA AUGMENTATION (On-the-fly)
# Otimizado para Google Colab GPU
# Foco: Detecção de Arritmias (N, S, V)
# ===================================================================

import os
import numpy as np
import pickle
import tensorflow as tf
from google.colab import drive
drive.mount('/content/drive')
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuração de GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"✅ GPU disponível: {physical_devices[0]}")
else:
    print("⚠️ GPU não detectada, usando CPU")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ===================================================================
# FUNÇÃO DE AUGMENTATION
# ===================================================================

def augment_ecg(x_cnn, noise_level=0.03, sampling_rate=500):
    """Aplica augmentation em um sinal ECG"""
    augmented = x_cnn

    # 1. Ruído gaussiano (60% chance)
    if np.random.rand() > 0.4:
        noise = np.random.normal(0, noise_level, x_cnn.shape)
        augmented = augmented + noise

    # 2. Scaling (60% chance)
    if np.random.rand() > 0.4:
        scale = np.random.uniform(0.9, 1.1)
        augmented = augmented * scale

    # 3. Baseline wander (50% chance)
    if np.random.rand() > 0.5:
        baseline_freq = np.random.uniform(0.1, 0.5)
        t = np.arange(len(x_cnn)) / sampling_rate
        baseline = 0.1 * np.sin(2 * np.pi * baseline_freq * t)
        augmented = augmented + baseline[:, np.newaxis]

    return augmented.astype(np.float32)


# ===================================================================
# MODELO CNN 1D HÍBRIDO
# ===================================================================

def build_cnn1d_hybrid_model(input_shape_cnn=(5000, 12),
                             input_shape_temporal=(10,),
                             num_classes=3,
                             dropout_rate=0.5):
    """
    Modelo CNN 1D híbrido com branch temporal
    Arquitetura multi-escala otimizada para arritmias
    """

    # === BRANCH 1: CNN 1D para sinais ECG ===
    input_cnn = layers.Input(shape=input_shape_cnn, name='ecg_signal')

    # Bloco 1: Detecção de ondas locais (P, QRS, T)
    x1 = layers.Conv1D(64, kernel_size=7, padding='same', name='conv1_local')(input_cnn)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)  # 5000 → 2500
    x1 = layers.Dropout(0.3)(x1)

    # Bloco 2: Padrões intermediários (segmentos ST)
    x2 = layers.Conv1D(128, kernel_size=15, padding='same', name='conv2_intermediate')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.MaxPooling1D(pool_size=2)(x2)  # 2500 → 1250
    x2 = layers.Dropout(0.3)(x2)

    # Bloco 3: Ritmo global (intervalos RR)
    x3 = layers.Conv1D(256, kernel_size=31, padding='same', name='conv3_global')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.MaxPooling1D(pool_size=2)(x3)  # 1250 → 625
    x3 = layers.Dropout(0.3)(x3)

    # Bloco 4: Features de alto nível
    x4 = layers.Conv1D(512, kernel_size=15, padding='same', name='conv4_high')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)
    x4_global = layers.GlobalMaxPooling1D()(x4)

    # Pooling alternativo para mais features
    x4_avg = layers.GlobalAveragePooling1D()(x4)

    # === BRANCH 2: Features temporais (HRV, etc) ===
    input_temporal = layers.Input(shape=input_shape_temporal, name='temporal_features')

    x_temp = layers.Dense(64, activation='relu')(input_temporal)
    x_temp = layers.BatchNormalization()(x_temp)
    x_temp = layers.Dropout(0.3)(x_temp)

    x_temp = layers.Dense(32, activation='relu')(x_temp)
    x_temp = layers.BatchNormalization()(x_temp)

    # === FUSÃO DOS BRANCHES ===
    combined = layers.Concatenate()([x4_global, x4_avg, x_temp])

    # === CLASSIFICADOR ===
    x = layers.Dense(512, activation='relu', name='fc1')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.8)(x)

    x = layers.Dense(128, activation='relu', name='fc3')(x)
    x = layers.Dropout(dropout_rate * 0.6)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # === MODELO FINAL ===
    model = Model(inputs=[input_cnn, input_temporal], outputs=outputs,
                  name='CNN1D_Hybrid_Arrhythmia')

    return model


# ===================================================================
# FUNÇÕES DE AVALIAÇÃO
# ===================================================================

def plot_training_history(history, save_path='./results/'):
    """Plota histórico de treinamento"""
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Acurácia
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_model(model, dataset, y_true, class_names, save_path='./results/'):
    """Avalia modelo e gera relatórios"""
    os.makedirs(save_path, exist_ok=True)

    # Predições
    logging.info("Gerando predições...")
    y_pred_probs = model.predict(dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    logging.info("\n=== CLASSIFICATION REPORT ===")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ROC-AUC (se multiclasse)
    if len(class_names) > 2:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            auc_scores = []

            for i in range(len(class_names)):
                auc = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
                auc_scores.append(auc)
                logging.info(f"AUC {class_names[i]}: {auc:.4f}")

            logging.info(f"Average AUC: {np.mean(auc_scores):.4f}")
        except Exception as e:
            logging.warning(f"Erro ao calcular AUC: {e}")

    return y_pred, y_pred_probs


# ===================================================================
# SCRIPT PRINCIPAL DE TREINAMENTO
# ===================================================================

if __name__ == "__main__":

    # Configurações
    DATA_PATH = '/content/drive/MyDrive/ptbxl_cnn1d_preprocessed.npz'
    METADATA_PATH = './preprocessing_metadata.pkl'
    SAVE_MODEL_PATH = './models/'
    RESULTS_PATH = './results/'
    DRIVE_MODEL_PATH = '/content/drive/MyDrive/models_arrhythmia/'

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Hiperparâmetros
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5

    # ===================================================================
    # 1. CARREGAMENTO DOS DADOS
    # ===================================================================

    logging.info("=== CARREGANDO DADOS PRÉ-PROCESSADOS ===")

    data = np.load(DATA_PATH, allow_pickle=True)

    X_train_cnn = data['X_train_cnn']
    X_train_temporal = data['X_train_temporal']
    y_train = data['y_train']

    X_val_cnn = data['X_val_cnn']
    X_val_temporal = data['X_val_temporal']
    y_val = data['y_val']

    X_test_cnn = data['X_test_cnn']
    X_test_temporal = data['X_test_temporal']
    y_test = data['y_test']

    label_encoder = data['label_encoder'].item()
    class_names = label_encoder.classes_

    logging.info(f"Treino: {len(X_train_cnn)} amostras")
    logging.info(f"Validação: {len(X_val_cnn)} amostras")
    logging.info(f"Teste: {len(X_test_cnn)} amostras")
    logging.info(f"Classes: {class_names}")
    logging.info(f"Shape CNN: {X_train_cnn.shape}")
    logging.info(f"Shape Temporal: {X_train_temporal.shape}")

    # ===================================================================
    # 2. CÁLCULO DE CLASS WEIGHTS
    # ===================================================================

    logging.info("\n=== CALCULANDO CLASS WEIGHTS ===")

    class_weights_values = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights_values))

    for i, (class_name, weight) in enumerate(zip(class_names, class_weights_values)):
        logging.info(f"Classe {class_name}: weight = {weight:.2f}")

    # ===================================================================
    # 3. CRIAÇÃO DE TF.DATA.DATASET
    # ===================================================================

    logging.info("\n=== CRIANDO TF.DATA.DATASETS ===")

    def create_dataset(X_cnn, X_temporal, y, batch_size, augment=False, shuffle=False):
      """Cria tf.data.Dataset otimizado"""

      def py_augment(x_cnn, x_temp, y_label):
          """Wrapper para augmentation em numpy"""
          x_cnn_np = x_cnn.numpy()
          x_temp_np = x_temp.numpy()
          y_label_np = y_label.numpy()

          if augment and np.random.rand() > 0.3:  # 70% de chance
              x_cnn_np = augment_ecg(x_cnn_np)

          return x_cnn_np.astype(np.float32), x_temp_np.astype(np.float32), y_label_np.astype(np.int64)

      def tf_augment(x_cnn, x_temp, y_label):
          """Wrapper TensorFlow para py_function"""
          x_cnn_aug, x_temp_aug, y_aug = tf.py_function(
              func=py_augment,
              inp=[x_cnn, x_temp, y_label],
              Tout=[tf.float32, tf.float32, tf.int64]
          )

          # Define shapes explicitamente
          x_cnn_aug.set_shape([5000, 12])
          x_temp_aug.set_shape(x_temp.shape)
          y_aug.set_shape([])

          return (x_cnn_aug, x_temp_aug), y_aug

      # Cria dataset base
      dataset = tf.data.Dataset.from_tensor_slices(((X_cnn, X_temporal), y))

      if shuffle:
          dataset = dataset.shuffle(buffer_size=min(len(X_cnn), 10000))

      if augment:
          # Desempacota a tupla antes de mapear
          dataset = dataset.map(
              lambda inputs, label: tf_augment(inputs[0], inputs[1], label),
              num_parallel_calls=tf.data.AUTOTUNE
          )

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(tf.data.AUTOTUNE)

      return dataset

    # Cria datasets
    train_dataset = create_dataset(
        X_train_cnn.astype(np.float32),
        X_train_temporal.astype(np.float32),
        y_train.astype(np.int64),
        batch_size=BATCH_SIZE,
        augment=True,
        shuffle=True
    )

    val_dataset = create_dataset(
        X_val_cnn.astype(np.float32),
        X_val_temporal.astype(np.float32),
        y_val.astype(np.int64),
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )

    test_dataset = create_dataset(
        X_test_cnn.astype(np.float32),
        X_test_temporal.astype(np.float32),
        y_test.astype(np.int64),
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )

    logging.info(f"Train dataset criado")
    logging.info(f"Val dataset criado")
    logging.info(f"Test dataset criado")

    # ===================================================================
    # 4. CONSTRUÇÃO DO MODELO
    # ===================================================================

    logging.info("\n=== CONSTRUINDO MODELO CNN 1D HÍBRIDO ===")

    model = build_cnn1d_hybrid_model(
        input_shape_cnn=(5000, 12),
        input_shape_temporal=(X_train_temporal.shape[1],),
        num_classes=len(class_names),
        dropout_rate=DROPOUT_RATE
    )

    model.summary()

    # ===================================================================
    # 5. COMPILAÇÃO DO MODELO
    # ===================================================================

    logging.info("\n=== COMPILANDO MODELO ===")

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ===================================================================
    # 6. CALLBACKS
    # ===================================================================

    logging.info("\n=== CONFIGURANDO CALLBACKS ===")

    callbacks_list = [
        # ModelCheckpoint - salva melhor modelo
        callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_MODEL_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # EarlyStopping - para se não melhorar
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        # ReduceLROnPlateau - reduz learning rate
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),

        # CSVLogger - salva histórico
        callbacks.CSVLogger(
            filename=os.path.join(RESULTS_PATH, 'training_log.csv'),
            separator=',',
            append=False
        )
    ]

    # ===================================================================
    # 7. TREINAMENTO
    # ===================================================================

    logging.info("\n=== INICIANDO TREINAMENTO ===")
    logging.info(f"Épocas: {EPOCHS}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Learning rate: {LEARNING_RATE}")
    logging.info(f"Data augmentation: ON (tf.data)")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    # ===================================================================
    # 8. VISUALIZAÇÃO DO TREINAMENTO
    # ===================================================================

    logging.info("\n=== PLOTANDO HISTÓRICO DE TREINAMENTO ===")
    plot_training_history(history, save_path=RESULTS_PATH)

    # ===================================================================
    # 9. AVALIAÇÃO NO CONJUNTO DE TESTE
    # ===================================================================

    logging.info("\n=== AVALIANDO NO CONJUNTO DE TESTE ===")

    # Carrega melhor modelo
    best_model = keras.models.load_model(
        os.path.join(SAVE_MODEL_PATH, 'best_model.h5')
    )

    y_pred, y_pred_probs = evaluate_model(
        best_model, test_dataset, y_test, class_names, save_path=RESULTS_PATH
    )

    # ===================================================================
    # 10. SALVA RESULTADOS FINAIS
    # ===================================================================

    logging.info("\n=== SALVANDO RESULTADOS FINAIS ===")

    model_production_path = os.path.join(DRIVE_MODEL_PATH, 'model_production.h5')
    best_model.save(model_production_path)
    logging.info(f"✅ Modelo salvo: {model_production_path}")

    # ✅ 2. LABEL ENCODER + METADADOS MÍNIMOS
    production_metadata = {
        'label_encoder': label_encoder,  # Para decodificar classes (N, S, V)
        'class_names': class_names.tolist(),  # ['N', 'S', 'V']
        'input_shape_cnn': (5000, 12),  # Shape esperado do sinal
        'input_shape_temporal': X_train_temporal.shape[1],  # Número de features temporais
        'sampling_rate': 500  # Hz
    }

    metadata_production_path = os.path.join(DRIVE_MODEL_PATH, 'model_metadata.pkl')
    with open(metadata_production_path, 'wb') as f:
        pickle.dump(production_metadata, f)
    logging.info(f"✅ Metadados salvos: {metadata_production_path}")

    logging.info(f"\nModelos salvos em: {SAVE_MODEL_PATH}")
    logging.info(f"Resultados salvos em: {RESULTS_PATH}")
    logging.info("\n✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")