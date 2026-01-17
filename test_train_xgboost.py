"""
Script simples para testar a função train_xgboost
Pode ser executado diretamente: python test_train_xgboost.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import xgboost as xgb
from src.modeling import (
    load_and_prepare_data,
    encode_categorical_features,
    prepare_features_and_target,
    split_data,
    train_xgboost
)

def test_with_synthetic_data():
    """Testa train_xgboost com dados sintéticos (rápido)"""
    print("="*80)
    print("TESTE 1: Dados Sintéticos")
    print("="*80)
    
    # Criar dados sintéticos
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Criar target binário desbalanceado (80% classe 0, 20% classe 1)
    y = pd.Series([0] * 800 + [1] * 200, dtype='Int64')
    
    # Split dos dados
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.05, random_state=42)
    
    print(f"Dados de treino: {len(X_train)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")
    print(f"Distribuição treino - Classe 0: {(y_train == 0).sum()}, Classe 1: {(y_train == 1).sum()}")
    print(f"Distribuição teste - Classe 0: {(y_test == 0).sum()}, Classe 1: {(y_test == 1).sum()}")
    
    # Treinar modelo
    print("\nTreinando modelo...")
    try:
        model, history = train_xgboost(X_train, y_train, X_test, y_test, focus_precision=True)
        
        print("\n✓ Modelo treinado com sucesso!")
        print(f"Best iteration: {history['best_iteration']}")
        print(f"Best score (logloss): {history['best_score']:.4f}")
        print(f"Scale pos weight: {history['scale_pos_weight']:.3f}")
        if 'optimal_threshold' in history:
            print(f"Optimal threshold: {history['optimal_threshold']:.3f}")
        
        # Fazer predições para testar
        # Usar os mesmos nomes de features que foram usados no treinamento
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
        dtest = xgb.DMatrix(X_test.values, feature_names=feature_names)
        predictions = model.predict(dtest)
        # Usar threshold ótimo se disponível, senão usar 0.5
        threshold = history.get('optimal_threshold', 0.5)
        predictions_binary = (predictions >= threshold).astype(int)
        
        print(f"\nPredições geradas: {len(predictions)}")
        print(f"Classe 0 prevista: {(predictions_binary == 0).sum()}")
        print(f"Classe 1 prevista: {(predictions_binary == 1).sum()}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Erro ao treinar modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_data():
    """Testa train_xgboost com dados reais do projeto"""
    print("\n" + "="*80)
    print("TESTE 2: Dados Reais do Projeto")
    print("="*80)
    
    csv_path = 'data/raw/Task_Exception Prediction_Training Test Data.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ Arquivo não encontrado: {csv_path}")
        print("   Pulando teste com dados reais...")
        return False
    
    try:
        # Carregar e preparar dados
        print("\n1. Carregando e limpando dados...")
        df_cleaned = load_and_prepare_data(csv_path, verbose=False)
        
        print("2. Codificando features categóricas...")
        df_encoded, encoders = encode_categorical_features(df_cleaned, verbose=False)
        
        print("3. Preparando features e target...")
        X, y = prepare_features_and_target(df_encoded, verbose=False)
        
        print("4. Dividindo em treino e teste...")
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.05, random_state=42)
        
        print(f"\nDados de treino: {len(X_train)} amostras, {X_train.shape[1]} features")
        print(f"Dados de teste: {len(X_test)} amostras")
        print(f"Distribuição treino - Classe 0: {(y_train == 0).sum()}, Classe 1: {(y_train == 1).sum()}")
        print(f"Distribuição teste - Classe 0: {(y_test == 0).sum()}, Classe 1: {(y_test == 1).sum()}")
        
        # Treinar modelo
        print("\n5. Treinando modelo XGBoost...")
        model, history = train_xgboost(X_train, y_train, X_test, y_test, focus_precision=True)
        
        print("\n✓ Modelo treinado com sucesso!")
        print(f"Best iteration: {history['best_iteration']}")
        print(f"Best score (logloss): {history['best_score']:.4f}")
        print(f"Scale pos weight: {history['scale_pos_weight']:.3f}")
        if 'optimal_threshold' in history:
            print(f"Optimal threshold: {history['optimal_threshold']:.3f}")
        
        # Fazer predições para testar
        # Limpar nomes das features como feito no treinamento
        feature_names_clean = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                               for col in X_test.columns]
        dtest = xgb.DMatrix(X_test.values, feature_names=feature_names_clean)
        predictions = model.predict(dtest)
        # Usar threshold ótimo se disponível, senão usar 0.5
        threshold = history.get('optimal_threshold', 0.5)
        predictions_binary = (predictions >= threshold).astype(int)
        
        print(f"\nPredições geradas: {len(predictions)}")
        print(f"Classe 0 prevista: {(predictions_binary == 0).sum()}")
        print(f"Classe 1 prevista: {(predictions_binary == 1).sum()}")
        
        # Calcular métricas básicas
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, predictions_binary)
        precision = precision_score(y_test, predictions_binary, zero_division=0)
        recall = recall_score(y_test, predictions_binary, zero_division=0)
        
        print(f"\nMétricas de teste:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        if precision > 0.5:
            print(f"\n✓ Precision > 50%: {precision:.2%} (Requisito do case atendido!)")
        else:
            print(f"\n⚠ Precision < 50%: {precision:.2%} (Pode precisar ajustar hiperparâmetros)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTE DA FUNÇÃO train_xgboost")
    print("="*80 + "\n")
    
    # Teste 1: Dados sintéticos (sempre funciona)
    success1 = test_with_synthetic_data()
    
    # Teste 2: Dados reais (requer arquivo CSV)
    success2 = test_with_real_data()
    
    # Resumo
    print("\n" + "="*80)
    print("RESUMO DOS TESTES")
    print("="*80)
    print(f"Teste 1 (Dados Sintéticos): {'✓ PASSOU' if success1 else '✗ FALHOU'}")
    print(f"Teste 2 (Dados Reais): {'✓ PASSOU' if success2 else '✗ FALHOU ou não executado'}")
    print("="*80 + "\n")
