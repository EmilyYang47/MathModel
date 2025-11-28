# Math Expression Solver Language Model

This repository contains a class project for **DS542: Deep Learning for Data Science**.

## Project Description

In this project, we built and trained a language model capable of solving mathematical expressions composed of positive integers, addition operations, and parentheses. The model takes expressions as input and outputs step-by-step evaluated results.

## Files

- `INPUT.txt` — Input file with math expressions   
- `math.pt` — Trained model weights  
- `predict.py` — Script to load the model and generate solutions for input expressions  
- `project3.ipynb` — Jupyter notebook detailing the model design, training, and evaluation  

## Usage

Run the prediction script with:

```bash
python3 predict.py INPUT.txt
```

This command reads expressions from `INPUT.txt`, generates and prints the completed solutions. 
