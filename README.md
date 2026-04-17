Sistema de Manutencao Preditiva com Deep Learning
Preveja falhas em maquinas industriais antes que elas acontecam e reduza custos de parada nao planejada.

https://img.shields.io/badge/Python-3.9+-blue.svg
https://img.shields.io/badge/TensorFlow-2.x-orange.svg
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/LinkedIn-Connect-blue.svg

O Problema Que Este Projeto Resolve
Em industrias, paradas nao planejadas custam milhoes. O desafio e identificar quando uma maquina vai falhar antes que o problema aconteca, usando apenas os dados dos sensores.

Este sistema detecta 97% das falhas com apenas 2 falsos negativos a cada 2.000 operacoes.

Resultados de Negocio
Metrica	Resultado	Impacto
Recall (Falhas Detectadas)	97.1%	Apenas 2 em 68 falhas passariam despercebidas
Falsos Positivos	0	Nenhum alarme falso – equipe de manutencao nao e sobrecarregada
AUC-ROC	0.977	Modelo tem excelente poder de discriminacao
Threshold Otimizado	0.3	Prioriza encontrar falhas (em vez de evitar alarmes falsos)
Traducao para o negocio: Para cada 100 falhas reais, o sistema alerta sobre 97 delas com tempo suficiente para manutencao preventiva.

Visualizacoes do Modelo
https://visualizacoes.png

O que cada grafico demonstra:

Grafico	O que mostra
Curva de Aprendizado	O modelo aprendeu corretamente (curvas de treino e validacao estao proximas)
Distribuicao das Probabilidades	Falhas (vermelho) e operacoes normais (verde) sao bem separadas
Curva ROC	AUC de 0.977 indica excelente capacidade de discriminacao
Matriz de Confusao	97% das falhas detectadas com ZERO falsos positivos
Arquitetura do Modelo
Camada	Detalhes
Entrada	12 features
Camada Dense 1	32 neuronios, ReLU, L2
Dropout 1	50%
Camada Dense 2	16 neuronios, ReLU, L2
Dropout 2	50%
Saida	Sigmoid (probabilidade de falha)
Por que esta arquitetura?

Regularizacao L2 → Evita overfitting (dados de treino vs. dados reais)

Dropout 50% → Forca o modelo a aprender padroes robustos

Sigmoid na saida → Probabilidade entre 0 e 1, ideal para decisoes de negocio

Tecnicas Aplicadas
Tratamento de dados desbalanceados (classe de falha recebe peso 14.76x maior)

Early Stopping para evitar overfitting e economizar tempo de treino

Otimizacao de threshold baseada no custo de negocio (priorizar recall sobre precisao)

Normalizacao com StandardScaler

Visualizacoes profissionais

Como Executar
Pre-requisitos
pip install -r requirements.txt

Executar o treinamento
python treinar_modelo.py

Usar o modelo para prever novas falhas
from demo import prever_falha

resultado = prever_falha(
temperatura_ar=298.5,
temperatura_processo=309.2,
rotacao=1450,
torque=45.3,
desgaste_ferramenta=120
)
print(resultado)

Dataset utilizado: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)

Estrutura do Projeto
├── treinar_modelo.py # Script principal de treinamento
├── demo.py # Script de demonstracao
├── requirements.txt # Dependencias do projeto
├── modelo_falhas.h5 # Modelo treinado (gerado)
├── scaler.pkl # Normalizador salvo (gerado)
├── visualizacoes.png # Graficos do modelo (gerado)
└── README.md # Este arquivo

Como Me Contratar para um Projeto Similar
Servico	Prazo	Investimento
Adaptar este modelo para os dados da SUA fabrica	5 dias	R$ 1.200 – R$ 2.500
Criar um dashboard de monitoramento em tempo real	3 dias	R$ 800 – R$ 1.500
Consultoria para estruturar dados de sensores	2 dias	R$ 600 – R$ 1.000
Me encontre no Workana ou Upwork – me chame pelo nome "Ivan Santos"

LinkedIn: https://www.linkedin.com/in/ivan-santos-8046a8355/

Habilidades Demonstradas
Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)

Redes Neurais para classificacao binaria

Tratamento de dados desbalanceados

Visualizacao de dados (Matplotlib)

Business acumen: escolha de metricas e threshold baseado em custo

Licenca
MIT – use a vontade, mas me de os creditos se for comercializar.

Deixe uma estrela se este projeto te ajudou!

