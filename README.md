# IDS basado en Transformer para Detección de Ataques DDoS

Este proyecto implementa un **Sistema de Detección de Intrusos (IDS)** utilizando un **modelo Transformer** aplicado a series temporales de tráfico de red. El objetivo es clasificar ventanas de tráfico como **BENIGN** o **ATTACK (DDoS)** a partir de datos del dataset CIC-IDS.

---

## 📌 Características principales

- Modelo **Transformer** con *Multi-Head Attention*
- Codificación **posicional sinusoidal**
- Agregación temporal mediante **attention-weighted pooling**
- Pipeline completo:
  - Limpieza de datos
  - Normalización
  - Generación de secuencias temporales
  - División Train / Validation / Test con *shuffle*
- Manejo de **desbalance de clases** con `class_weight`
- Selección automática del **mejor threshold** usando F1-score
- Visualización de métricas y matriz de confusión

---

## Estructura del proyecto:

```text
.
├── model.py                      # Script principal (este código)
├── scaler.pkl                    # Scaler entrenado (se genera automáticamente)
├── ids_transformer_model.h5      # Modelo entrenado
├── training_history.png          # Gráfica de entrenamiento
└── README.md
````

## Configuración del modelo:

````json
CONFIG = {
    'sequence_length': 100,
    'step_size': 80,
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
````
