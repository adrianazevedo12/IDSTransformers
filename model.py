import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

# Configuración Global
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONFIG = {
    'csv_path': "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 
    'sequence_length': 100,
    'step_size': 80,
    'feature_dim': None,
    'batch_size': 32,
    'epochs': 180,
    'learning_rate': 0.00005,
    'model': {
        'num_heads': 4,
        'hidden_dim': 24,
        'dropout_rate': 0.15,
        'l2_reg': 0.001,
        'num_blocks': 2
    }
}

################################################################################
#                               MODELO TRANSFORMER
################################################################################

class CyberSecurityTransformer:
    def __init__(self, sequence_length, feature_dim, num_heads=4, hidden_dim=24,
                 dropout_rate=0.15, l2_reg=0.001, num_blocks=2):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.num_blocks = num_blocks
        self.num_heads = self._adjust_num_heads(num_heads, feature_dim)

        print(f"[MODEL] Inicializando Transformer (Heads: {self.num_heads}, Dim: {feature_dim})")
        self.model = self.build_model()

    def _adjust_num_heads(self, requested_heads, feature_dim):
        valid_heads = [h for h in range(1, min(requested_heads + 1, feature_dim + 1)) 
                       if feature_dim % h == 0]
        if not valid_heads: return 1
        return min(max(valid_heads), 4)

    def get_positional_encoding(self):
        position = np.arange(self.sequence_length)[:, np.newaxis]
        freq_sincos = np.exp(np.arange(0, self.feature_dim, 2) * -(np.log(10000.0) / self.feature_dim))
        pos_encoding = np.zeros((self.sequence_length, self.feature_dim))
        pos_encoding[:, 0::2] = np.sin(position * freq_sincos)
        if self.feature_dim % 2 != 0:
            pos_encoding[:, 1::2] = np.cos(position * freq_sincos[:-1])
        else:
            pos_encoding[:, 1::2] = np.cos(position * freq_sincos)
        pos_encoding = pos_encoding / np.sqrt(self.feature_dim)
        return tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def transformer_block(self, inputs, block_id):
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.feature_dim // self.num_heads,
            dropout=self.dropout_rate,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(inputs, inputs)
        attention = tf.keras.layers.Dropout(self.dropout_rate)(attention)
        # Aquí usamos Add() porque ambos son tensores de Keras, aquí sí funciona
        attention = tf.keras.layers.Add()([inputs, attention])
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)

        ff = tf.keras.layers.Dense(self.hidden_dim, activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(attention)
        ff = tf.keras.layers.Dropout(self.dropout_rate)(ff)
        ff = tf.keras.layers.Dense(self.feature_dim, 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(ff)
        ff = tf.keras.layers.Dropout(self.dropout_rate)(ff)
        
        output = tf.keras.layers.Add()([attention, ff])
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(output)

    def attention_weighted_aggregation(self, x):
        scores = tf.keras.layers.Dense(1, activation='tanh')(x)
        weights = tf.keras.layers.Softmax(axis=1)(scores)
        weighted_features = tf.keras.layers.Multiply()([x, weights])
        return tf.keras.layers.Lambda(lambda inputs: tf.reduce_sum(inputs, axis=1))(weighted_features)

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        scaled_inputs = inputs * tf.math.sqrt(tf.cast(self.feature_dim, tf.float32))
        pos_encoding = self.get_positional_encoding()
        x = scaled_inputs + pos_encoding

        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        for i in range(self.num_blocks):
            x = self.transformer_block(x, i)

        x = self.attention_weighted_aggregation(x)
        x = tf.keras.layers.Dense(self.hidden_dim, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CyberSecurityTransformer')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'], clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model

################################################################################
#                               PIPELINE DE DATOS
################################################################################

class DataPipeline:
    @staticmethod
    def load_clean_data(file_path):
        print(f"[DATA] Cargando: {file_path}")
        if not os.path.exists(file_path): raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        df = pd.read_csv(file_path, sep=None, engine='python', thousands=None)
        
        # Corrección nombre columna Label
        if 'Label' not in df.columns:
            posibles = [c for c in df.columns if 'Label' in c]
            if posibles: df.rename(columns={posibles[0]: 'Label'}, inplace=True)
            else: raise ValueError("Columna 'Label' no encontrada")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' not in numeric_cols: numeric_cols.append('Label')
        df = df[numeric_cols]
        
        df['Label'] = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' or x == 0 else 1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        print(f"[DATA] Limpio: {df.shape} | Ataques: {df['Label'].sum()}")
        return df

    @staticmethod
    def normalize_data(df):
        feature_cols = [c for c in df.columns if c != 'Label']
        X = df[feature_cols].values
        y = df['Label'].values
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'scaler.pkl')
        return X_scaled, y

    @staticmethod
    def create_sequences(data, labels, seq_len, step):
        sequences, seq_labels = [], []
        for i in range(0, len(data) - seq_len + 1, step):
            window = data[i:i + seq_len]
            label_window = labels[i:i + seq_len]
            # Etiqueta: 1 si hay ataque en el 20% final de la ventana
            if np.sum(label_window[-int(seq_len*0.2):]) > 0:
                sequences.append(window)
                seq_labels.append(1)
            else:
                sequences.append(window)
                seq_labels.append(0)
        return np.array(sequences), np.array(seq_labels)

    @staticmethod
    def split_data_shuffled(X_sequences, y_sequences, train_ratio=0.7, val_ratio=0.10):
        """
        División con SHUFFLE (Barajado) para asegurar distribución de clases.
        """
        n_sequences = len(X_sequences)
        print(f"[SPLIT] Realizando división con SHUFFLE (Total: {n_sequences})")

        # 1. Barajado aleatorio (TU LÓGICA ORIGINAL)
        np.random.seed(42)
        indices = np.arange(n_sequences)
        np.random.shuffle(indices)

        X_sequences = X_sequences[indices]
        y_sequences = y_sequences[indices]

        # 2. Calcular puntos de corte
        train_end = int(n_sequences * train_ratio)
        val_end = int(n_sequences * (train_ratio + val_ratio))

        # 3. División
        X_train, y_train = X_sequences[:train_end], y_sequences[:train_end]
        X_val, y_val = X_sequences[train_end:val_end], y_sequences[train_end:val_end]
        X_test, y_test = X_sequences[val_end:], y_sequences[val_end:]

        # 4. Validaciones de seguridad
        sets = {'Train': y_train, 'Val': y_val, 'Test': y_test}
        valid_split = True
        
        for name, y_set in sets.items():
            if len(y_set) > 0:
                n_attacks = np.sum(y_set == 1)
                pct = (n_attacks / len(y_set)) * 100
                print(f"   > {name}: {n_attacks} ataques ({pct:.2f}%)")
                
                if n_attacks == 0 and name != 'Train':
                    print(f"[ERROR] {name} no contiene ataques.")
                    valid_split = False
            else:
                 print(f"[ERROR] Set {name} vacío.")

        if not valid_split:
            print("[CRITICAL] Dataset no apto. Ajusta el ratio o cambia el CSV.")
            return None, None, None, None, None, None

        print(f"[SUCCESS] División completada correctamente.")
        return X_train, y_train, X_val, y_val, X_test, y_test

################################################################################
#                               GRAFICAS
################################################################################

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['loss'], label='Train'); ax[0].plot(history.history['val_loss'], label='Val')
    ax[0].set_title('Loss'); ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='Train'); ax[1].plot(history.history['val_accuracy'], label='Val')
    ax[1].set_title('Accuracy'); ax[1].legend()
    plt.tight_layout(); plt.savefig('training_history.png'); plt.show()

def evaluate_model(model, X_test, y_test):
    print("\n[EVAL] Evaluando modelo...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    
    print(f"Mejor Threshold: {best_threshold:.4f}")
    print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Thr={best_threshold:.2f})'); plt.show()

################################################################################
#                               MAIN
################################################################################

def main():
    print("=== INICIANDO SISTEMA IDS TRANSFORMER (MODO SHUFFLE) ===")
    
    try:
        df = DataPipeline.load_clean_data(CONFIG['csv_path'])
    except Exception as e:
        print(f"[ERROR] {e}"); return

    X_scaled, y = DataPipeline.normalize_data(df)
    CONFIG['feature_dim'] = X_scaled.shape[1]

    print("[INFO] Generando secuencias...")
    X_seq, y_seq = DataPipeline.create_sequences(
        X_scaled, y, CONFIG['sequence_length'], CONFIG['step_size']
    )

    X_train, y_train, X_val, y_val, X_test, y_test = DataPipeline.split_data_shuffled(X_seq, y_seq)
    
    if X_train is None: return # Error en split

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"[INFO] Class Weights: {class_weight_dict}")

    transformer = CyberSecurityTransformer(
        sequence_length=CONFIG['sequence_length'],
        feature_dim=CONFIG['feature_dim'],
        **CONFIG['model']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    print("[TRAIN] Iniciando entrenamiento...")
    history = transformer.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    transformer.model.save('ids_transformer_model.h5')
    plot_history(history)
    evaluate_model(transformer.model, X_test, y_test)
    print("[SUCCESS] Pipeline finalizado.")

if __name__ == '__main__':
    main()