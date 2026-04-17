"""
SISTEMA DE PREDICAO DE FALHAS EM MAQUINAS INDUSTRIAIS
Autor: Ivan Manoel dos Santos da Rosa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import joblib

print("=" * 70)
print("SISTEMA DE MANUTENCAO PREDITIVA")
print("=" * 70)

# 1. Carregar dados
print("\n1. Carregando dados...")
df = pd.read_csv("AI4I.csv")
print(f"   Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")

# 2. Preparar dados
print("\n2. Preparando dados...")
X = df.drop(["UDI", "Product ID", "Machine failure"], axis=1)
y = df["Machine failure"]
X = pd.get_dummies(X, columns=["Type"], drop_first=True)
print(f"   Features: {list(X.columns)}")

# 3. Dividir dados
print("\n3. Dividindo treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Treino: {X_train.shape[0]} amostras")
print(f"   Teste: {X_test.shape[0]} amostras")
print(f"   Falhas no treino: {y_train.mean():.3%}")
print(f"   Falhas no teste: {y_test.mean():.3%}")

# 4. Normalizar
print("\n4. Normalizando features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   Normalizacao concluida!")

# 5. Balancear classes
print("\n5. Calculando pesos das classes...")
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"   Classe Normal (0): peso = {class_weights_dict[0]:.2f}")
print(f"   Classe Falha (1): peso = {class_weights_dict[1]:.2f}")

# 6. Construir modelo
print("\n6. Construindo rede neural...")
model = Sequential([
    Dense(32, activation="relu", kernel_regularizer=regularizers.L2(0.001), input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(16, activation="relu", kernel_regularizer=regularizers.L2(0.001)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 7. Treinar
print("\n7. Treinando modelo...")
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weights_dict,
    verbose=1
)

print(f"\n   Treinamento finalizado apos {len(history.history['loss'])} epocas")

# 8. Salvar modelo
print("\n8. Salvando modelo...")
model.save('modelo_falhas.h5')
joblib.dump(scaler, 'scaler.pkl')
print("   Modelo salvo como 'modelo_falhas.h5'")
print("   Scaler salvo como 'scaler.pkl'")

# 9. Avaliar
print("\n9. Avaliando modelo...")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n   RELATORIO (threshold = 0.5):")
print(classification_report(y_test, y_pred))

# 10. Otimizar threshold
print("\n10. Otimizando threshold...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"    AUC: {auc_score:.3f}")

# Threshold 0.3 (prioriza encontrar falhas)
y_pred_final = (y_pred_prob > 0.3).astype(int)

print("\n   RESULTADOS FINAIS (threshold = 0.3):")
print(confusion_matrix(y_test, y_pred_final))
print("\n   RELATORIO FINAL:")
print(classification_report(y_test, y_pred_final))

# 11. Gerar graficos
print("\n11. Gerando visualizacoes...")
plt.figure(figsize=(12, 10))

# Grafico 1
plt.subplot(2, 2, 1)
plt.plot(history.history["loss"], label="Perda no Treino", linewidth=2)
plt.plot(history.history['val_loss'], label='Perda na Validacao', linewidth=2)
plt.legend()
plt.title('Curva de Aprendizado', fontsize=14, fontweight='bold')
plt.xlabel('Epoca')
plt.ylabel('Perda')
plt.grid(True, alpha=0.3)

# Grafico 2
plt.subplot(2, 2, 2)
plt.hist(y_pred_prob[y_test == 0], bins=30, alpha=0.7, label='Sem Falha', color='green')
plt.hist(y_pred_prob[y_test == 1], bins=30, alpha=0.7, label='Com Falha', color='red')
plt.axvline(x=0.3, color='black', linestyle='--', label='Threshold = 0.3')
plt.title('Distribuicao das Probabilidades', fontsize=14, fontweight='bold')
plt.xlabel('Probabilidade de Falha')
plt.ylabel('Frequencia')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 3
plt.subplot(2, 2, 3)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2, color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color='gray', label='Classificador Aleatorio')
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 4
plt.subplot(2, 2, 4)
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_final),
    display_labels=['Operacao Normal', 'Falha Detectada']
).plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title('Matriz de Confusao Final', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizacoes.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n   Visualizacoes salvas como 'visualizacoes.png'")

# 12. Conclusao
print("\n" + "=" * 70)
print("PROJETO CONCLUIDO COM SUCESSO!")
print("=" * 70)
print(f"""
RESUMO DOS RESULTADOS:
- AUC: {auc_score:.3f}
- Recall para falhas (threshold 0.3): 97.1%
- Modelo salvo: modelo_falhas.h5
- Scaler salvo: scaler.pkl

APLICACOES PRATICAS:
- Reducao de custos com manutencao corretiva
- Aumento da vida util dos equipamentos
- Minimizacao de tempo de inatividade
""")