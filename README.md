# Sistema de Predição de Falhas em Máquinas Industriais

## 📝 Visão Geral do Projeto
Este projeto implementa um sistema de **manutenção preditiva** para identificar e prever falhas em máquinas industriais. Utilizando uma **Rede Neural Artificial (RNA)** construída com **TensorFlow/Keras**, o modelo analisa dados de sensores para prever a probabilidade de falha, permitindo intervenções proativas e reduzindo custos operacionais.

---

## 👤 Autor
**Ivan Manoel dos Santos da Rosa**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white )](https://www.linkedin.com/in/ivan-santos-8046a8355/ )

---

## 📊 Dataset
O projeto utiliza o `Predictive Maintenance Dataset (AI4I.csv)`, que contém dados reais de sensores de máquinas industriais, incluindo temperatura, velocidade de rotação e torque.

---

## 🛠️ Metodologia

### 1. Tecnologias Utilizadas
*   **Linguagem**: Python
*   **Manipulação de Dados**: Pandas e NumPy
*   **Visualização**: Matplotlib e Seaborn
*   **Machine Learning**: Scikit-learn
*   **Deep Learning**: TensorFlow e Keras

### 2. Pré-processamento dos Dados
*   **Seleção de Features**: Remoção de identificadores irrelevantes.
*   **Codificação**: Aplicação de One-Hot Encoding para variáveis categóricas.
*   **Normalização**: Uso de `StandardScaler` para otimizar o aprendizado da rede neural.
*   **Balanceamento**: Tratamento de classes desbalanceadas através de `class_weights`.

### 3. Arquitetura da Rede Neural
*   **Camadas Ocultas**: Camadas densas com ativação `ReLU`.
*   **Regularização**: Implementação de `Dropout` e `L2` para evitar overfitting.
*   **Otimização**: Uso do otimizador `Adam` e `EarlyStopping`.

---

## 🚀 Resultados e Avaliação
O projeto foca na métrica de **Recall**, essencial para manutenção preditiva. Com o threshold ajustado para **0.3**, o modelo alcançou:
*   **Alta detecção de falhas reais**, minimizando paradas não planejadas.
*   **Excelente poder discriminativo** validado via Curva ROC/AUC.
*   **Matriz de Confusão** otimizada para cenários industriais críticos.

---

## ⚙️ Como Executar
1. Clone este repositório.
2. Instale as dependências:  
   `pip install pandas numpy matplotlib scikit-learn tensorflow`
3. Abra o arquivo `.ipynb` no Google Colab ou Jupyter Notebook.
