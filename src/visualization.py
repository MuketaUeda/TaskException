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
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular porcentagens
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plotar matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    # Adicionar porcentagens
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                          ha="center", va="center", color="red", fontsize=10)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 8)):
    """
    Plota gráfico de barras com as features mais importantes do modelo.
    
    Args:
        model: Modelo XGBoost treinado
        feature_names: Lista com nomes das features
        top_n: Número de top features para mostrar (default: 15)
        figsize: Tamanho da figura (default: (10, 8))
    """
    try:
        from src.modeling import get_feature_importance
    except ImportError:
        from modeling import get_feature_importance
    
    # Obter importância das features
    df_importance = get_feature_importance(model, feature_names, top_n=top_n)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar barras horizontais
    colors = sns.color_palette("husl", len(df_importance))
    bars = ax.barh(range(len(df_importance)), df_importance['importance'], color=colors)
    
    # Configurar eixos
    ax.set_yticks(range(len(df_importance)))
    ax.set_yticklabels(df_importance['feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {len(df_importance)} Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Mostrar maior importância no topo
    
    # Adicionar valores nas barras
    for i, (idx, row) in enumerate(df_importance.iterrows()):
        ax.text(row['importance'], i, f' {row["importance"]:.0f}',
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


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
    metrics = ['accuracy', 'precision', 'recall']
    train_values = [metrics_train.get(m, 0) for m in metrics]
    test_values = [metrics_test.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Metrics Comparison: Train vs Test', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Treino
    train_counts = pd.Series(y_train).value_counts().sort_index()
    train_percent = train_counts / len(y_train) * 100
    
    bars1 = ax1.bar(train_counts.index.astype(str), train_counts.values, 
                    color=['green', 'red'], alpha=0.7)
    ax1.set_xlabel('Class', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Training Set Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Adicionar valores e porcentagens
    for i, (idx, count) in enumerate(train_counts.items()):
        ax1.text(i, count, f'{count}\n({train_percent[idx]:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Teste
    test_counts = pd.Series(y_test).value_counts().sort_index()
    test_percent = test_counts / len(y_test) * 100
    
    bars2 = ax2.bar(test_counts.index.astype(str), test_counts.values,
                    color=['green', 'red'], alpha=0.7)
    ax2.set_xlabel('Class', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Test Set Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Adicionar valores e porcentagens
    for i, (idx, count) in enumerate(test_counts.items()):
        ax2.text(i, count, f'{count}\n({test_percent[idx]:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


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
    try:
        from src.modeling import check_overfitting
    except ImportError:
        from modeling import check_overfitting
    
    # Verificar overfitting
    metrics = {'train': metrics_train, 'test': metrics_test}
    overfitting_info = check_overfitting(metrics)
    
    metrics_names = ['accuracy', 'precision', 'recall']
    train_values = [metrics_train.get(m, 0) for m in metrics_names]
    test_values = [metrics_test.get(m, 0) for m in metrics_names]
    gaps = [train - test for train, test in zip(train_values, test_values)]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Gráfico 1: Comparação de métricas
    bars1 = ax1.bar(x - width/2, train_values, width, label='Train', alpha=0.8, color='blue')
    bars2 = ax1.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='orange')
    
    ax1.set_xlabel('Metrics', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Train vs Test Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in metrics_names])
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Gap (diferença)
    colors = ['red' if abs(g) > 0.1 else 'orange' if abs(g) > 0.05 else 'green' 
              for g in gaps]
    bars3 = ax2.bar(metrics_names, gaps, color=colors, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
    ax2.axhline(y=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Metrics', fontsize=11)
    ax2.set_ylabel('Gap (Train - Test)', fontsize=11)
    ax2.set_title(f'Overfitting Analysis\nSeverity: {overfitting_info["severity"].upper()}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xticklabels([m.capitalize() for m in metrics_names])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.3f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 8)):
    """
    Plota curva ROC (Receiver Operating Characteristic).
    
    Mostra performance do modelo em diferentes thresholds.
    
    Args:
        y_true: Valores reais do target
        y_pred_proba: Probabilidades preditas pelo modelo (probabilidade de classe 1)
        figsize: Tamanho da figura (default: (8, 8))
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar curva ROC
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


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
    try:
        from src.modeling import evaluate_model, get_feature_importance, check_overfitting
    except ImportError:
        from modeling import evaluate_model, get_feature_importance, check_overfitting
    
    import xgboost as xgb
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Obter threshold das métricas
    threshold = metrics_train.get('threshold', 0.5) if isinstance(metrics_train, dict) and 'threshold' in metrics_train else 0.5
    
    # Fazer predições
    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
        feature_names_clean = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                              for col in X_test.columns]
    else:
        X_test_array = X_test
        feature_names_clean = feature_names
    
    dtest = xgb.DMatrix(X_test_array, feature_names=feature_names_clean)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Converter y_test para array se necessário
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.values
    else:
        y_test_array = y_test
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Matriz de Confusão
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test_array, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['GOOD', 'RED'], yticklabels=['GOOD', 'RED'],
                cbar_kws={'label': 'Count'}, ax=ax1)
    for i in range(2):
        for j in range(2):
            ax1.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                    ha="center", va="center", color="red", fontsize=9)
    ax1.set_xlabel('Predicted', fontsize=10)
    ax1.set_ylabel('Actual', fontsize=10)
    ax1.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
    
    # 2. Feature Importance
    ax2 = fig.add_subplot(gs[0, 1])
    df_importance = get_feature_importance(model, feature_names, top_n=10)
    colors = sns.color_palette("husl", len(df_importance))
    ax2.barh(range(len(df_importance)), df_importance['importance'], color=colors)
    ax2.set_yticks(range(len(df_importance)))
    ax2.set_yticklabels(df_importance['feature'], fontsize=8)
    ax2.set_xlabel('Importance', fontsize=10)
    ax2.set_title('Top 10 Feature Importance', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. Comparação de Métricas
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['accuracy', 'precision', 'recall']
    train_vals = [metrics_train.get(m, 0) for m in metrics]
    test_vals = [metrics_test.get(m, 0) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
    ax3.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
    ax3.set_xlabel('Metrics', fontsize=10)
    ax3.set_ylabel('Score', fontsize=10)
    ax3.set_title('Metrics Comparison', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.capitalize() for m in metrics])
    ax3.legend()
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Distribuição de Classes - Treino
    ax4 = fig.add_subplot(gs[1, 0])
    train_counts = pd.Series(y_train).value_counts().sort_index()
    train_percent = train_counts / len(y_train) * 100
    ax4.bar(train_counts.index.astype(str), train_counts.values, 
            color=['green', 'red'], alpha=0.7)
    for i, (idx, count) in enumerate(train_counts.items()):
        ax4.text(i, count, f'{count}\n({train_percent[idx]:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax4.set_xlabel('Class', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Training Set Distribution', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Distribuição de Classes - Teste
    ax5 = fig.add_subplot(gs[1, 1])
    test_counts = pd.Series(y_test).value_counts().sort_index()
    test_percent = test_counts / len(y_test) * 100
    ax5.bar(test_counts.index.astype(str), test_counts.values,
            color=['green', 'red'], alpha=0.7)
    for i, (idx, count) in enumerate(test_counts.items()):
        ax5.text(i, count, f'{count}\n({test_percent[idx]:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax5.set_xlabel('Class', fontsize=10)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('Test Set Distribution', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Análise de Overfitting
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_dict = {'train': metrics_train, 'test': metrics_test}
    overfitting_info = check_overfitting(metrics_dict)
    gaps = [train_vals[i] - test_vals[i] for i in range(len(metrics))]
    colors_gap = ['red' if abs(g) > 0.1 else 'orange' if abs(g) > 0.05 else 'green' 
                  for g in gaps]
    ax6.bar(metrics, gaps, color=colors_gap, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax6.axhline(y=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Metrics', fontsize=10)
    ax6.set_ylabel('Gap (Train - Test)', fontsize=10)
    ax6.set_title(f'Overfitting Analysis\n({overfitting_info["severity"]})', 
                  fontsize=11, fontweight='bold')
    ax6.set_xticklabels([m.capitalize() for m in metrics])
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. ROC Curve
    ax7 = fig.add_subplot(gs[2, :])
    fpr, tpr, _ = roc_curve(y_test_array, y_pred_proba)
    roc_auc = roc_auc_score(y_test_array, y_pred_proba)
    ax7.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax7.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random (AUC = 0.500)')
    ax7.set_xlim([0.0, 1.0])
    ax7.set_ylim([0.0, 1.05])
    ax7.set_xlabel('False Positive Rate', fontsize=11)
    ax7.set_ylabel('True Positive Rate', fontsize=11)
    ax7.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax7.legend(loc="lower right", fontsize=10)
    ax7.grid(alpha=0.3)
    
    # Título geral
    fig.suptitle('Complete Model Report', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Report saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
