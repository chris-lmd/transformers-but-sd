# -*- coding: utf-8 -*-
"""
Fonctions utilitaires pour les TP Transformers
BUT SD 3ème année - Module Réseaux de Neurones Approfondissement
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List


# =============================================================================
# VISUALISATION ATTENTION
# =============================================================================

def plot_attention_weights(attention_weights: torch.Tensor,
                           tokens_x: Optional[List[str]] = None,
                           tokens_y: Optional[List[str]] = None,
                           title: str = "Poids d'attention",
                           figsize: tuple = (8, 6)):
    """
    Affiche une heatmap des poids d'attention.

    Args:
        attention_weights: Tensor de shape (seq_len, seq_len) ou (batch, seq, seq)
        tokens_x: Labels pour l'axe X (tokens clés)
        tokens_y: Labels pour l'axe Y (tokens requêtes)
        title: Titre du graphique
        figsize: Taille de la figure
    """
    # Si batch dimension, prendre le premier élément
    if attention_weights.dim() == 3:
        attention_weights = attention_weights[0]

    weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap='Blues')

    # Labels
    if tokens_x is not None:
        ax.set_xticks(range(len(tokens_x)))
        ax.set_xticklabels(tokens_x, rotation=45, ha='right')
    if tokens_y is not None:
        ax.set_yticks(range(len(tokens_y)))
        ax.set_yticklabels(tokens_y)

    ax.set_xlabel("Clés (Keys)")
    ax.set_ylabel("Requêtes (Queries)")
    ax.set_title(title)

    # Colorbar
    plt.colorbar(im, ax=ax)

    # Valeurs dans les cellules
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            text = ax.text(j, i, f'{weights[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if weights[i, j] > 0.5 else "black",
                          fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_multihead_attention(attention_weights: torch.Tensor,
                             num_heads: int,
                             tokens: Optional[List[str]] = None,
                             figsize: tuple = (15, 4)):
    """
    Affiche les poids d'attention pour chaque tête.

    Args:
        attention_weights: Tensor de shape (batch, num_heads, seq, seq)
        num_heads: Nombre de têtes d'attention
        tokens: Labels des tokens
        figsize: Taille de la figure
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Premier élément du batch

    fig, axes = plt.subplots(1, num_heads, figsize=figsize)

    for head in range(num_heads):
        weights = attention_weights[head].detach().cpu().numpy()

        ax = axes[head] if num_heads > 1 else axes
        im = ax.imshow(weights, cmap='Blues')
        ax.set_title(f'Tête {head + 1}')

        if tokens is not None:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=8)

    plt.suptitle("Attention par tête", fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# POSITIONAL ENCODING - VISUALISATION
# =============================================================================

def plot_positional_encoding(pe: torch.Tensor,
                             max_display: int = 50,
                             figsize: tuple = (12, 6)):
    """
    Visualise le positional encoding.

    Args:
        pe: Tensor de positional encoding (1, seq_len, embed_dim)
        max_display: Nombre max de positions à afficher
        figsize: Taille de la figure
    """
    if pe.dim() == 3:
        pe = pe[0]

    pe_np = pe[:max_display].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap
    im = axes[0].imshow(pe_np.T, cmap='RdBu', aspect='auto')
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("Positional Encoding (Heatmap)")
    plt.colorbar(im, ax=axes[0])

    # Courbes pour quelques dimensions
    dims_to_plot = [0, 1, 2, 3, 10, 20]
    for d in dims_to_plot:
        if d < pe_np.shape[1]:
            axes[1].plot(pe_np[:, d], label=f'dim {d}')
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Valeur")
    axes[1].set_title("Positional Encoding (Courbes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MÉTRIQUES ET ÉVALUATION
# =============================================================================

def plot_training_history(train_losses: List[float],
                          val_losses: Optional[List[float]] = None,
                          train_accs: Optional[List[float]] = None,
                          val_accs: Optional[List[float]] = None,
                          figsize: tuple = (12, 4)):
    """
    Affiche les courbes d'entraînement.
    """
    num_plots = 1 + (train_accs is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    if num_plots == 1:
        axes = [axes]

    # Loss
    axes[0].plot(train_losses, label='Train Loss')
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if train_accs is not None:
        axes[1].plot(train_accs, label='Train Acc')
        if val_accs:
            axes[1].plot(val_accs, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calcule l'accuracy.
    """
    preds = predictions.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


# =============================================================================
# UTILITAIRES TEXTE
# =============================================================================

def simple_tokenizer(text: str, vocab: dict) -> List[int]:
    """
    Tokenizer simple pour les exercices.

    Args:
        text: Texte à tokenizer
        vocab: Dictionnaire mot -> index

    Returns:
        Liste d'indices
    """
    tokens = text.lower().split()
    return [vocab.get(t, vocab.get('<unk>', 0)) for t in tokens]


def create_vocab(texts: List[str], min_freq: int = 1) -> dict:
    """
    Crée un vocabulaire à partir d'une liste de textes.
    """
    word_counts = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {'<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3}
    for word, count in sorted(word_counts.items()):
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


# =============================================================================
# UTILITAIRES MODÈLES
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Compte le nombre de paramètres d'un modèle.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_shape: tuple = None):
    """
    Affiche un résumé du modèle.
    """
    print(f"{'='*60}")
    print(f"Modèle: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Paramètres entraînables: {count_parameters(model):,}")
    print(f"{'='*60}")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    print(f"{'='*60}")


# =============================================================================
# DONNÉES EXEMPLE
# =============================================================================

def get_sample_sentences_fr():
    """
    Retourne des phrases exemple en français pour les exercices.
    """
    return [
        "Le chat mange la souris",
        "La souris mange le fromage",
        "Le chien court dans le jardin",
        "Le jardin est très grand",
        "La maison est belle",
    ]


def get_sample_sentences_en():
    """
    Retourne des phrases exemple en anglais pour les exercices.
    """
    return [
        "The cat eats the mouse",
        "The mouse eats the cheese",
        "The dog runs in the garden",
        "The garden is very big",
        "The house is beautiful",
    ]