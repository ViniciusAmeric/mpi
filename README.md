# Multiplicação de Matrizes com MPI (mpi4py)

## Descrição

Este projeto implementa a multiplicação de matrizes quadradas utilizando computação paralela com MPI (Message Passing Interface), através da biblioteca mpi4py em Python.

O objetivo é demonstrar como dividir um problema computacionalmente intensivo entre múltiplos processos executando em paralelo no mesmo processador (multiprocessamento local).

---

## Estratégia Utilizada

1. O processo raiz (rank 0) gera duas matrizes quadradas N x N.
2. A matriz A é dividida por linhas e distribuída entre os processos utilizando Scatter.
3. A matriz B é enviada completa para todos os processos utilizando Broadcast.
4. Cada processo realiza a multiplicação da sua parte local (np.dot).
5. Os resultados parciais são reunidos no processo raiz utilizando Gather.
6. O tempo de execução paralela é medido com MPI.Wtime().

Esse modelo é conhecido como "Row-wise Parallel Matrix Multiplication".

---

## Requisitos

- Python 3
- OpenMPI instalado
- mpi4py
- numpy

Instalar dependências:

pip install mpi4py numpy

---

## Como Executar

Ativar ambiente virtual (opcional):

source .venv/bin/activate

Executar com 4 processos:

mpirun -np 4 python matriz_mpi.py

Importante:
O tamanho da matriz (N) deve ser divisível pelo número de processos.

---

## Conceitos Demonstrados

- Computação Paralela
- Multiprocessamento
- Comunicação distribuída
- MPI (Scatter, Broadcast, Gather)
- Medição de tempo paralelo
- Divisão de carga por blocos de linhas

---

## Observação

Para matrizes pequenas (ex: 4x4), o ganho de desempenho pode não ser perceptível devido ao custo de comunicação entre processos.
Para matrizes maiores, a paralelização tende a trazer melhor desempenho.