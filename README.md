# Sistema de Manutencao Preditiva com Deep Learning

**Preveja falhas em maquinas industriais antes que elas acontecam e reduza custos de parada nao planejada.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## O Problema Que Este Projeto Resolve

Em industrias, **paradas nao planejadas custam milhoes**. O desafio e identificar **quando uma maquina vai falhar** antes que o problema aconteca, usando apenas os dados dos sensores.

**Este sistema detecta 97% das falhas** com apenas 2 falsos negativos a cada 2.000 operacoes.

---

## Resultados de Negocio (O que voce ganha)

| Metrica | Resultado | Impacto |
|:--------|:----------|:--------|
| **Recall (Falhas Detectadas)** | **97.1%** | Apenas 2 em 68 falhas passariam despercebidas |
| **Falsos Positivos** | 0 | Nenhum alarme falso – equipe de manutencao nao e sobrecarregada |
| **AUC-ROC** | 0.974 | Modelo tem excelente poder de discriminacao |
| **Threshold Otimizado** | 0.3 | Prioriza encontrar falhas (em vez de evitar alarmes falsos) |

> **Traducao para o negocio:** Para cada 100 falhas reais, o sistema alerta sobre 97 delas com tempo suficiente para manutencao preventiva.

---

## Arquitetura do Modelo

```
Entrada (12 features)
    ↓
Camada Dense (32 neuronios, ReLU + L2)
    ↓
Dropout (50%)
    ↓
Camada Dense (16 neuronios, ReLU + L2)
    ↓
Dropout (50%)
    ↓
Saida Sigmoid (probabilidade de falha)
```

**Por que esta arquitetura?**
- **Regularizacao L2** → Evita overfitting (dados de treino vs. dados reais)
- **Dropout 50%** → Forca o modelo a aprender padroes robustos
- **Sigmoid na saida** → Probabilidade entre 0 e 1, ideal para decisoes de negocio

---

## Tecnicas Aplicadas (O que mostro que sei fazer)

- Tratamento de **dados desbalanceados** (pesos de classe – classe de falha recebe peso 14.76x maior)
- **Early Stopping** para evitar overfitting e economizar tempo de treino
- **Otimizacao de threshold** baseada no custo de negocio (priorizar recall sobre precisao)
- Normalizacao com StandardScaler
- Visualizacoes profissionais: Curva de aprendizado, matriz de confusao, distribuicao de probabilidades, curva ROC

---

## Visualizacoes Geradas

*(Execute o notebook para gerar os graficos abaixo)*

| Curva de Aprendizado | Distribuicao de Probabilidades |
|:--------------------:|:------------------------------:|
| (inserir imagem learning_curve.png) | (inserir imagem prob_distribution.png) |

| Curva ROC | Matriz de Confusao |
|:---------:|:------------------:|
| (inserir imagem roc_curve.png) | (inserir imagem confusion_matrix.png) |

---

## Como Executar

```bash
# Clone o repositorio
git clone https://github.com/santos-design/predicao-falhas-industriais-deep-learning.git

# Instale as dependencias
pip install -r requirements.txt

# Execute o notebook
jupyter notebook predicao_falhas.ipynb
```

**Dataset utilizado:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)

---

## Como Me Contratar para um Projeto Similar

| Servico | Prazo | Investimento |
|:--------|:------|:-------------|
| Adaptar este modelo para os dados da SUA fabrica | 5 dias | R$ 1.200 – R$ 2.500 |
| Criar um dashboard de monitoramento em tempo real (Streamlit/Power BI) | 3 dias | R$ 800 – R$ 1.500 |
| Consultoria para estruturar dados de sensores e definir KPIs de manutencao | 2 dias | R$ 600 – R$ 1.000 |

**Me encontre no Workana ou Upwork** – me chame pelo nome "Ivan Santos"

---

## Habilidades Demonstradas

- Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)
- Redes Neurais para classificacao binaria
- Tratamento de dados desbalanceados (class_weight)
- Visualizacao de dados (Matplotlib, ConfusionMatrixDisplay)
- Business acumen: escolha de metricas (Recall > Acurácia) e threshold baseado em custo

---

## Licenca

MIT – use a vontade, mas me de os creditos se for comercializar.

---

**Deixe uma estrela se este projeto te ajudou!**
