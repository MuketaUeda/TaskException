import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, labels=['GOOD', 'RED'], title='Confusion Matrix'):
    """
    Plota matriz de confusão com valores e porcentagens.
    
    Args:
        y_true: Valores reais do target
        y_pred: Valores preditos pelo modelo
        labels: Labels das classes (default: ['GOOD', 'RED'])
        title: Título do gráfico
    """
    pass


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 8)):
    """
    Plota gráfico de barras com as features mais importantes do modelo.
    
    Args:
        model: Modelo XGBoost treinado
        feature_names: Lista com nomes das features
        top_n: Número de top features para mostrar (default: 15)
        figsize: Tamanho da figura (default: (10, 8))
    """
    pass


def plot_metrics_comparison(metrics_train, metrics_test, figsize=(12, 4)):
    """
    Plota comparação de métricas entre treino e teste.
    
    Métricas comparadas:
    - Accuracy
    - Precision
    - Recall
    
    Args:
        metrics_train: Dict com métricas de treino {'accuracy': ..., 'precision': ..., 'recall': ...}
        metrics_test: Dict com métricas de teste {'accuracy': ..., 'precision': ..., 'recall': ...}
        figsize: Tamanho da figura (default: (12, 4))
    """
    pass


def plot_class_distribution(y_train, y_test, figsize=(12, 4)):
    """
    Plota distribuição das classes em treino e teste.
    
    Mostra:
    - Contagem absoluta de cada classe
    - Porcentagem de cada classe
    
    Args:
        y_train: Target do conjunto de treino
        y_test: Target do conjunto de teste
        figsize: Tamanho da figura (default: (12, 4))
    """
    pass


def plot_overfitting_analysis(metrics_train, metrics_test, figsize=(10, 6)):
    """
    Plota análise visual de overfitting comparando métricas de treino vs teste.
    
    Mostra:
    - Gap entre métricas de treino e teste
    - Indicação se há overfitting baseado no gap
    
    Args:
        metrics_train: Dict com métricas de treino
        metrics_test: Dict com métricas de teste
        figsize: Tamanho da figura (default: (10, 6))
    """
    pass


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 8)):
    """
    Plota curva ROC (Receiver Operating Characteristic).
    
    Mostra performance do modelo em diferentes thresholds.
    
    Args:
        y_true: Valores reais do target
        y_pred_proba: Probabilidades preditas pelo modelo (probabilidade de classe 1)
        figsize: Tamanho da figura (default: (8, 8))
    """
    pass


def create_model_report(model, X_train, y_train, X_test, y_test, feature_names, 
                       metrics_train, metrics_test, save_path=None):
    """
    Cria relatório visual completo do modelo com todos os gráficos.
    
    Inclui:
    - Matriz de confusão
    - Feature importance
    - Comparação de métricas (train vs test)
    - Distribuição de classes
    - Análise de overfitting
    
    Args:
        model: Modelo XGBoost treinado
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        feature_names: Lista com nomes das features
        metrics_train: Dict com métricas de treino
        metrics_test: Dict com métricas de teste
        save_path: Caminho para salvar figura (default: None, apenas mostra)
    """
    pass
