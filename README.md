Sistema de Predição de Falhas em Máquinas Industriais
Visão Geral do Projeto
Este projeto implementa um sistema de manutenção preditiva para identificar e prever falhas em máquinas industriais. Utilizando uma Rede Neural Artificial (RNA) construída com TensorFlow/Keras, o modelo analisa dados de sensores para prever a probabilidade de falha, permitindo intervenções proativas e reduzindo custos operacionais.
Autor
Ivan Manoel dos Santos da Rosa
LinkedIn
Dataset
O projeto utiliza o Predictive Maintenance Dataset (AI4I.csv), que contém dados reais de sensores de máquinas industriais, incluindo temperatura, velocidade de rotação e torque.
Metodologia
1. Tecnologias Utilizadas
Linguagem: Python
Manipulação de Dados: Pandas e NumPy
Visualização: Matplotlib e Seaborn
Machine Learning: Scikit-learn
Deep Learning: TensorFlow e Keras
2. Pré-processamento dos Dados
Seleção de Features: Remoção de identificadores irrelevantes para o modelo.
Codificação: Aplicação de One-Hot Encoding para variáveis categóricas.
Normalização: Uso de StandardScaler para garantir que todas as features estejam na mesma escala, otimizando o aprendizado da rede neural.
Balanceamento: Tratamento de classes desbalanceadas através do cálculo de pesos de classe (class_weights), garantindo que o modelo aprenda a identificar falhas mesmo sendo eventos raros.
3. Arquitetura da Rede Neural
O modelo foi desenvolvido com uma estrutura sequencial profunda:
Camadas Ocultas: Camadas densas com ativação ReLU.
Regularização: Implementação de Dropout e regularização L2 para evitar overfitting e garantir a generalização para novos dados.
Otimização: Uso do otimizador Adam e função de perda binary_crossentropy.
Callbacks: Utilização de EarlyStopping para interromper o treinamento no ponto ideal de performance.
Resultados e Avaliação
O projeto foca na métrica de Recall, essencial para manutenção preditiva. Através da otimização do threshold de classificação (ajustado para 0.3), o modelo alcançou:
Alta detecção de falhas reais, minimizando o risco de paradas não planejadas.
Análise via Curva ROC/AUC, demonstrando excelente poder discriminativo.
Matriz de Confusão, validando a eficácia do modelo em cenários industriais críticos.
Como Executar
Clone este repositório.
Certifique-se de ter as bibliotecas instaladas: pip install pandas numpy matplotlib scikit-learn tensorflow.
Abra o arquivo .ipynb no Google Colab ou Jupyter Notebook.
Execute as células para visualizar o treinamento e os gráficos de performance.

