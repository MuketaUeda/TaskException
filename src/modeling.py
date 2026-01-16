import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
import xgboost as xgb
from cleaning import NumericCleaner, TextNormalizer
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='data/raw/Task_Exception Prediction_Training Test Data.csv'):
    """
    Carrega o CSV e aplica os transformers de limpeza.
    
    Args:
        csv_path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame limpo e preparado
    """
    pass


def encode_categorical_features(df):
    """
    Aplica Label Encoding nas variáveis categóricas.
    
    Args:
        df: DataFrame com dados limpos
        
    Returns:
        DataFrame com categóricas encodadas e dict de encoders para referência
    """
    pass


def prepare_features_and_target(df):
    """
    Separa features (X) e target (y).
    
    Target: Exception_output
    - "RED" = 1 (exceção ocorreu)
    - "GOOD" = 0 (sem exceção)
    
    Args:
        df: DataFrame com dados encodados
        
    Returns:
        X (features), y (target)
    """
    pass


def split_data(X, y, test_size=0.05, random_state=42):
    """
    Divide dados em treino e teste.
    
    Requisito do case: 5% para teste, mantendo proporção das classes (stratified).
    
    Args:
        X: Features
        y: Target
        test_size: Proporção para teste (default 0.05 = 5%)
        random_state: Seed para reprodutibilidade
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    pass


def train_xgboost(X_train, y_train, X_test, y_test, focus_precision=True):
    """
    Treina modelo XGBoost focando em precisão.
    
    Requisito do case: Precisão >50% (priorizar precisão sobre recall).
    
    Args:
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste (para early stopping)
        focus_precision: Se True, ajusta para maximizar precisão
        
    Returns:
        Modelo treinado e histórico de treinamento
    """
    pass


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Avalia o modelo calculando métricas de performance.
    
    Métricas calculadas:
    - Accuracy: Proporção de predições corretas
    - Precision: Dos preditos como RED, quantos realmente são RED?
    - Recall: Dos RED reais, quantos foram preditos?
    
    Requisito do case: Focar em precisão >50%
    
    Args:
        model: Modelo treinado
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        
    Returns:
        Dict com métricas
    """
    pass


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extrai e mostra as features mais importantes do modelo.
    
    Args:
        model: Modelo XGBoost treinado
        feature_names: Lista com nomes das features
        top_n: Número de top features para mostrar
        
    Returns:
        DataFrame com features ordenadas por importância
    """
    pass


def check_overfitting(metrics):
    """
    Verifica se há overfitting comparando performance em treino vs teste.
    
    Args:
        metrics: Dict com métricas de treino e teste
    """
    pass
