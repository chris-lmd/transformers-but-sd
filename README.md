# Transformers - BUT Science des Données

Module **"Réseaux de neurones approfondissement"** - BUT SD 3ème année (Parcours EMS)

## Objectifs

Comprendre et implémenter l'architecture Transformer, la base de GPT, BERT, ChatGPT, etc.

## Prérequis

- Python (niveau intermédiaire)

## Programme

### Travaux Pratiques (12h)

| Session | Thème | Durée | Notebooks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|---------|-------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Fondamentaux NLP | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-01/TP-01-NLP-Fondamentaux.ipynb)                                                                                                                                                                                                                                                                                                                                                                      |
| 2 | Positional Encoding & Bases Attention | 2h | [![TP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-02/TP-02-PE_et_Bases_Attention.ipynb) [![Note](https://img.shields.io/badge/Note-PE-blue)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-02/Note-Positional-Encoding.ipynb)                                                                                                                                                                                           |
| 3 | Maîtriser l'Attention | 2h | [![TP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-03/TP-03-Attention.ipynb) [![Note](https://img.shields.io/badge/Note-Attention-blue)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-03/Note-Attention.ipynb)                                                                                                                                                                                                          |
| 4 | Multi-Head & Transformer | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-04/TP-04-MultiHead-Transformer.ipynb)                                                                                                                                                                                                                                                                                                                                                                 |
| 5 | Mini-GPT : Génération de noms | 2h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-05/TP-05-MiniGPT-Noms.ipynb)                                                                                                                                                                                                                                                                                                                                                                          |
| 6 | Fine-tuning GPT-2 | 2h | [![TP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-06/TP-06-MiniGPT-FineTuning.ipynb) [![Exploration](https://img.shields.io/badge/Exploration-Techniques-blue)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Session-06/TP-06-Exploration.ipynb) |

### Projets

| Projet | Description | Notebook |
|--------|-------------|----------|
| **Fake News** | Détecteur de fake news : From scratch + Fine-tuning DistilBERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Projet-FakeNews/TP-Projet-FakeNews.ipynb) |
| **Mini-GPT** | Génération de noms Fantasy avec décodeur + GPT-2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chris-lmd/transformers-but-sd/blob/main/Projet-MiniGPT/TP-Projet-MiniGPT.ipynb) |

## Progression pédagogique

```
Session 1: Tokenization → Embeddings → Word2Vec
              ↓
Session 2: Positional Encoding → Similarité → Q/K/V
              ↓
Session 3: scaled_dot_product_attention → SelfAttention → split_heads
              ↓
Session 4: concat_heads → MultiHeadAttention → FFN → Masque causal
              ↓
Session 5: TransformerBlock → Mini-GPT → Génération de noms Pokémon
              ↓
Session 6: Fine-tuning GPT-2 → Génération avancée
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
├── Session-01/
│   └── TP-01-NLP-Fondamentaux.ipynb
├── Session-02/
│   ├── TP-02-PE_et_Bases_Attention.ipynb
│   └── Note-Positional-Encoding.ipynb
├── Session-03/
│   ├── TP-03-Attention.ipynb
│   └── Note-Attention.ipynb
├── Session-04/
│   └── TP-04-MultiHead-Transformer.ipynb
├── Session-05/
│   └── TP-05-MiniGPT-Noms.ipynb
├── Session-06/
│   ├── TP-06-MiniGPT-FineTuning.ipynb
│   └── TP-06-Exploration.ipynb
├── Projet-FakeNews/
│   └── TP-Projet-FakeNews.ipynb
└── Projet-MiniGPT/
    └── TP-Projet-MiniGPT.ipynb
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