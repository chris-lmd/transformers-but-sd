# Transformers - BUT Science des Données

Module **"Réseaux de neurones approfondissement"** - BUT SD 3ème année (Parcours EMS)

## Objectifs

Comprendre et implémenter l'architecture Transformer, la base de GPT, BERT, ChatGPT, etc.

## Prérequis

- Python (niveau intermédiaire)
- Réseaux de neurones multicouches (MLP)
- CNN avec Keras/TensorFlow (notions)

## Sessions

| Session | Thème | Durée | Notebook |
|---------|-------|-------|----------|
| 1 | Mécanisme d'Attention | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/transformers-but-sd/blob/main/Session-01-Attention/TP-01-Attention.ipynb) |
| 2 | Multi-Head Attention | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/transformers-but-sd/blob/main/Session-02-MultiHead/TP-02-MultiHead.ipynb) |
| 3 | Architecture Transformer | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/transformers-but-sd/blob/main/Session-03-Transformer/TP-03-Transformer.ipynb) |
| 4 | Classification de Texte | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/transformers-but-sd/blob/main/Session-04-Classification/TP-04-Classification.ipynb) |
| 5 | Projet Final | 2h | Voir ci-dessous |

## Projet Final (Session 5)

Choisissez l'un des deux projets :

| Projet | Description | Notebook |
|--------|-------------|----------|
| **A - Fake News** | Détecteur de fake news FR/EN avec pipeline de traduction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/transformers-but-sd/blob/main/Session-05-Projet/Projet-A-FakeNews.ipynb) |
| **B - CLIP** | Recherche d'images par texte (multimodal) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/transformers-but-sd/blob/main/Session-05-Projet/Projet-B-CLIP.ipynb) |

## Installation (si usage local)

```bash
pip install torch torchvision transformers datasets matplotlib numpy
```

## Utilisation avec Google Colab (Recommandé)

1. Cliquez sur le badge "Open in Colab" du TP souhaité
2. Connectez-vous avec votre compte Google
3. Exécutez la première cellule pour installer les dépendances
4. C'est prêt !

## Structure du repo

```
transformers-but-sd/
├── Session-01-Attention/
│   └── TP-01-Attention.ipynb
├── Session-02-MultiHead/
│   └── TP-02-MultiHead.ipynb
├── Session-03-Transformer/
│   └── TP-03-Transformer.ipynb
├── Session-04-Classification/
│   └── TP-04-Classification.ipynb
├── Session-05-Projet/
│   ├── Projet-A-FakeNews.ipynb
│   └── Projet-B-CLIP.ipynb
├── utils/
│   └── helpers.py
└── data/
```

## Ressources

- [Documentation PyTorch](https://pytorch.org/docs/stable/)
- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [HuggingFace Documentation](https://huggingface.co/docs)

## Auteur

Module conçu pour le BUT Science des Données - IUT