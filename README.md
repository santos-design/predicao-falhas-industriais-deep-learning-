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

## Resultados de Negocio

| Metrica | Resultado | Impacto |
|:--------|:----------|:--------|
| **Recall (Falhas Detectadas)** | **97.1%** | Apenas 2 em 68 falhas passariam despercebidas |
| **Falsos Positivos** | 0 | Nenhum alarme falso – equipe de manutencao nao e sobrecarregada |
| **AUC-ROC** | 0.977 | Modelo tem excelente poder de discriminacao |
| **Threshold Otimizado** | 0.3 | Prioriza encontrar falhas (em vez de evitar alarmes falsos) |

> **Traducao para o negocio:** Para cada 100 falhas reais, o sistema alerta sobre 97 delas com tempo suficiente para manutencao preventiva.

---

## Visualizacoes do Modelo

![Visualizacoes do Modelo](visualizacoes.png)

**O que cada grafico demonstra:**

| Grafico | O que mostra |
|:--------|:-------------|
| **Curva de Aprendizado** | O modelo aprendeu corretamente (curvas de treino e validacao estao proximas) |
| **Distribuicao das Probabilidades** | Falhas (vermelho) e operacoes normais (verde) sao bem separadas |
| **Curva ROC** | AUC de 0.977 indica excelente capacidade de discriminacao |
| **Matriz de Confusao** | 97% das falhas detectadas com ZERO falsos positivos |

---

## Arquitetura do Modelo
