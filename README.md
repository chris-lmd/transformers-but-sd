# Transformers - BUT Science des Données

Module **"Réseaux de neurones approfondissement"** - BUT SD 3ème année (Parcours EMS)

## Objectifs

Comprendre et implémenter l'architecture Transformer, la base de GPT, BERT, ChatGPT, etc.

## Prérequis

- Python (niveau intermédiaire)

## Programme

### Travaux Pratiques (8h)

| TP | Thème | Durée | Notebook                                                                                                                                                                                                                |
|----|-------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Fondamentaux NLP | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-01-NLP-Fondamentaux/TP-01-NLP-Fondamentaux.ipynb) |
| 2 | Mécanisme d'Attention | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-02-Attention/TP-02-Attention.ipynb)               |
| 3 | Multi-Head Attention | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-03-MultiHead/TP-03-MultiHead.ipynb)               |
| 4 | Architecture Transformer | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-04-Transformer/TP-04-Transformer.ipynb)           |

### Projets (4h)

| Projet | Description | Durée | Notebook |
|--------|-------------|-------|----------|
| **Fake News** | Détecteur de fake news : From scratch + Fine-tuning DistilBERT | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Projet-FakeNews/TP-Projet-FakeNews.ipynb) |
| **Mini-GPT** | Génération de noms Fantasy avec décodeur + GPT-2 | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Projet-MiniGPT/TP-Projet-MiniGPT.ipynb) |

## Progression pédagogique

```
TP1: Tokenization → Embeddings → Word2Vec
           ↓
TP2: RNN/LSTM → Attention Q/K/V → Scaled Dot-Product
           ↓
TP3: split_heads → Multi-Head → concat_heads
           ↓
TP4: Positional Encoding → FFN → Residuals → LayerNorm
           ↓
Projet Fake News: Classification From Scratch + Fine-tuning
           ↓
Projet Mini-GPT: Décodeur + Masque causal + Génération
```

## Installation (si usage local)

```bash
pip install torch torchvision transformers datasets matplotlib numpy gensim scikit-learn tqdm
```

## Utilisation avec Google Colab (Recommandé)

1. Cliquez sur le badge "Open in Colab" du TP souhaité
2. Connectez-vous avec votre compte Google
3. Exécutez la première cellule pour installer les dépendances
4. C'est prêt !

## Structure du repo

```
transformers-but-sd/
├── Session-01-NLP-Fondamentaux/
│   └── TP-01-NLP-Fondamentaux.ipynb
├── Session-02-Attention/
│   └── TP-02-Attention.ipynb
├── Session-03-MultiHead/
│   └── TP-03-MultiHead.ipynb
├── Session-04-Transformer/
│   └── TP-04-Transformer.ipynb
├── Projet-FakeNews/
│   └── TP-Projet-FakeNews.ipynb
├── Projet-MiniGPT/
│   └── TP-Projet-MiniGPT.ipynb
├── utils/
└── data/
```

## Ressources

- [Documentation PyTorch](https://pytorch.org/docs/stable/)
- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Documentation](https://huggingface.co/docs)

## Auteurs
- [chris-lmd](https://github.com/chris-lmd)
- [ClementRx79](https://github.com/ClementRx79)
  
Module conçu pour le BUT Science des Données - IUT
