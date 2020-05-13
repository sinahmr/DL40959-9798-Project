# Image to Latex
An implementation of [What You Get Is What You See:
A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938v1.pdf) paper. This project was my Deep Learning course's project.

## Dataset
Dataset folder in this repository is a template.
You can download the dataset from [here](https://drive.google.com/file/d/12PPVScxRgMkraqXymy7szGKk4B7bMRQX/view) and place it in the `dataset` folder.

## Evaluation

#### Evaluating BLEU Score
```bash
python3 Evaluation/bleu_score.py --target-formulas target.txt --predicted-formulas predicted.txt --ngram 5
```

#### Evaluating Edit Distance Accuracy

```bash
python3 Evaluation/edit_distance.py --target-formulas target.txt --predicted-formulas predicted.txt
```
