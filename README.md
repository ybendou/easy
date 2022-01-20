# Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.
This repository is the official implementation Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.

EASY proposes a simple methodology, that reaches or even beats state of the art performance on multiple standardized benchmarks of the field, while adding almost no hyperparameters or parameters to those used for training the initial deep learning models on the generic dataset.

## Downloads 
Please click the Google Drive link for downloads:
- Datasets: link
- Features: link
- Models  : link

## Testing scripts for EASY
Run scripts to evaluate the features on FSL tasks for Y and ASY. For EY and EASY use the corresponding features.

### Inductive setup using NCM
Test features on miniimagenet using Y

    $ python main.py --dataset miniimagenet --model resnet12 --test-features "features path" --preprocessing ME

Test features on miniimagenet using ASY

    $ python main.py --dataset miniimagenet --model resnet12 --test-features "features path" --preprocessing ME --sample-aug 30


### Transductive setup using Soft k-means
Test features on miniimagenet using Y

    $ python main.py --dataset miniimagenet --model resnet12 --test-features "features path" --preprocessing ME --transductive --transductive-softkmeans --transductive-temperature-softkmeans 100

Test features on miniimagenet using ASY

    $ python main.py --dataset miniimagenet --model resnet12 --test-features "features path" --preprocessing ME --sample-aug 30 --transductive --transductive-softkmeans --transductive-temperature-softkmeans 100

## Training scripts for ASY
Train a model on miniimagenet using manifold mixup, self-supervision and cosine scheduler

    $ python main.py --dataset-path "dataset path" --dataset miniimagenet --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing ME 

## Important Arguments
Some important arguments for our code.

**Training arguments**
- `dataset`: choices=['miniimagenet', 'cubfs','tieredimagenet', 'fc100', 'cifarfs']
- `model`: choices=['resnet12', 'resnet18', 'resnet20', 'wideresnet', 's2m2r']

**Few-shot Classification**
- `preprocessing`: preprocessing sequence for few shot given as a string, can contain R:relu P:sqrt E:sphering and M:centering

## Few-shot classification Results

Experimental results on few-shot learning datasets with ResNet-12 backbone. We report our average results with 10000 randomly sampled episodes for both 1-shot and 5-shot evaluations.

**MiniImageNet Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
| Method 1 |     00.00    |     00.00    |
| method 2 |     **00.00**    |     **00.00**    |
| method 3  |     **00.00**    |     **00.00**    | 
| method 4 |     **00.00**    |     **00.00**    | 

**TieredImageNet Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
| Method 1 |     00.00    |     00.00    |
| method 2 |     **00.00**    |     **00.00**    |
| method 3  |     **00.00**    |     **00.00**    | 
| method 4 |     **00.00**    |     **00.00**    | 

**CUBFS Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
| Method 1 |     00.00    |     00.00    |
| method 2 |     **00.00**    |     **00.00**    |
| method 3  |     **00.00**    |     **00.00**    | 
| method 4 |     **00.00**    |     **00.00**    | 


**CIFAR-FS Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
| Method 1 |     00.00    |     00.00    |
| method 2 |     **00.00**    |     **00.00**    |
| method 3  |     **00.00**    |     **00.00**    | 
| method 4 |     **00.00**    |     **00.00**    | 

**FC-100 Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
| Method 1 |     00.00    |     00.00    |
| method 2 |     **00.00**    |     **00.00**    |
| method 3  |     **00.00**    |     **00.00**    | 
| method 4 |     **00.00**    |     **00.00**    | 

## Acknowledgment


