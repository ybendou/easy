# Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.
This repository is the official implementation Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.

EASY proposes a simple methodology, that reaches or even beats state of the art performance on multiple standardized benchmarks of the field, while adding almost no hyperparameters or parameters to those used for training the initial deep learning models on the generic dataset.

## Downloads 
Please click the [Google Drive link](https://drive.google.com/drive/folders/1fMeapvuR6Rby0HDHd5L74BEXRyiOF942) for downloading the features, backbones and datasets.

## Testing scripts for EASY
Run scripts to evaluate the features on FSL tasks for Y and ASY. For EY and EASY use the corresponding features.

### Inductive setup using NCM
Test features on miniimagenet using Y (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11' --preprocessing ME

Test features on miniimagenet using ASY (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11' --preprocessing ME --sample-aug 30

Test features on miniimagenet using EY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeatures1.pt11, <path>/minifeatures2.pt11, <path>/minifeatures3.pt11]" --preprocessing ME
    
Test features on miniimagenet using EASY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeatures1.pt11, <path>/minifeatures2.pt11, <path>/minifeatures3.pt11]" --preprocessing ME --sample-aug 30


### Transductive setup using Soft k-means
Test features on miniimagenet using Y (ResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11'--postprocessing ME --transductive --transductive-softkmeans --transductive-temperature-softkmeans 20

Test features on miniimagenet using ASY (ResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11' --postprocessing ME --sample-aug 30 --transductive --transductive-softkmeans --transductive-temperature-softkmeans 20

Test features on miniimagenet using EY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeatures1.pt11, <path>/minifeatures2.pt11, <path>/minifeatures3.pt11]" --postrocessing ME  --transductive --transductive-softkmeans --transductive-temperature-softkmeans 20

Test features on miniimagenet using EASY (3xResNet12)

    $ python main.py --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeatures1.pt11, <path>/minifeatures2.pt11, <path>/minifeatures3.pt11]" --postrocessing ME  --sample-aug 30 --transductive --transductive-softkmeans --transductive-temperature-softkmeans 20

## Training scripts for Y
Train a model on miniimagenet using manifold mixup, self-supervision and cosine scheduler

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing ME 

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
|SimpleShot [29] |62.85 ± 0.20 |80.02 ± 0.14|
|Baseline++ [30] |53.97 ± 0.79 |75.90 ± 0.61|
|TADAM [35] |58.50 ± 0.30| 76.70 ± 0.30|
|ProtoNet [10] |60.37 ± 0.83 |78.02 ± 0.57|
|R2-D2 (+ens) [20] |64.79 ± 0.45| 81.08 ± 0.32|
|FEAT [36] |66.78 |82.05|
|CNL [37] |67.96 ± 0.98 |83.36 ± 0.51|
|MERL [38] |67.40 ± 0.43 |83.40 ± 0.28|
|Deep EMD v2 [13] |68.77 ± 0.29 |84.13 ± 0.53|
|PAL [8] |69.37 ± 0.64 |84.40 ± 0.44|
|inv-equ [39] |67.28 ± 0.80 |84.78 ± 0.50|
|CSEI [40] |68.94 ± 0.28| 85.07 ± 0.50|
|COSOC [9] |69.28 ± 0.49| 85.16 ± 0.42|
|EASY 2×ResNet12 1/sqrt(2) (ours) |70.63 ± 0.20| 86.28 ± 0.12|
|:--------:|:------------:|:------------:|
|S2M2R [12] | 64.93 ± 0.18 | 83.18 ± 0.11 |
|LR + DC [17] | 68.55 ± 0.55 | 82.88 ± 0.42 |
|EASY 3×ResNet12 (ours) | 71.75 ± 0.19 | 87.15 ± 0.12 |

**TieredImageNet Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|SimpleShot [29]| 69.09 ± 0.22 |84.58 ± 0.16|
|ProtoNet [10]| 65.65 ± 0.92 |83.40 ± 0.65|
|FEAT [36] |70.80 ± 0.23| 84.79 ± 0.16|
|PAL [8]| 72.25 ± 0.72| 86.95 ± 0.47|
|DeepEMD v2 [13] |74.29 ± 0.32 |86.98 ± 0.60|
|MERL [38] |72.14 ± 0.51 |87.01 ± 0.35|
|COSOC [9] |73.57 ± 0.43 |87.57 ± 0.10|
|CNL [37]| 73.42 ± 0.95 |87.72 ± 0.75|
|invariance-equivariance [39] |72.21 ± 0.90| 87.08 ± 0.58|
|CSEI [40] |73.76 ± 0.32| 87.83 ± 0.59|
|ASY ResNet12 (ours)| 74.31 ± 0.22| 87.86 ± 0.15|
|:--------:|:------------:|:------------:|
|S2M2R [12] |73.71 ± 0.22| 88.52 ± 0.14|
|EASY 3×ResNet12 (ours) |74.71 ± 0.22| 88.33 ± 0.14|


**CUBFS Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|FEAT [36]| 68.87 ± 0.22| 82.90 ± 0.10|
|LaplacianShot [41]| 80.96 |88.68|
|ProtoNet [10] |66.09 ± 0.92 |82.50 ± 0.58|
|DeepEMD v2 [13] |79.27 ± 0.29| 89.80 ± 0.51|
|EASY 4×ResNet12 1/sqrt(2) |77.97 ± 0.20 | 91.59 ± 0.10|
|:--------:|:------------:|:------------:|
|S2M2R [12] | 80.68 ± 0.81 | 90.85 ± 0.44|
|EASY 3×ResNet12 (ours) | 78.56 ± 0.19 | 91.93 ± 0.10 |


**CIFAR-FS Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|S2M2R [12] |63.66 ± 0.17 | 76.07 ± 0.19|
|R2-D2 (+ens) [20] | 76.51 ± 0.47 | 87.63 ± 0.34|
|invariance-equivariance [39] | 77.87 ± 0.85 | 89.74 ± 0.57|
|EASY 2×ResNet12 1/sqrt(2) (ours) | 75.24 ± 0.20 | 88.38 ± 0.14|
|:--------:|:------------:|:------------:|
|S2M2R [12] |74.81 ± 0.19| 87.47 ± 0.13|
|EASY 3×ResNet12 (ours) | 76.20 ± 0.20 | 89.00 ± 0.14|

**FC-100 Dataset**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|DeepEMD v2 [13] | 46.60 ± 0.26 | 63.22 ± 0.71|
|TADAM [35] | 40.10 ± 0.40 | 56.10 ± 0.40|
|ProtoNet [10] | 41.54 ± 0.76 | 57.08 ± 0.76|
|invariance-equivariance [39] |47.76 ± 0.77 | 65.30 ± 0.76|
|R2-D2 (+ens) [20] | 44.75 ± 0.43 | 59.94 ± 0.41|
|EASY 2×ResNet12 1/sqrt(2) (ours)| 47.94 ± 0.19 | 64.14 ± 0.19|
|:--------:|:------------:|:------------:|
|EASY 3×ResNet12 (ours) | 48.07 ± 0.19 | 64.74 ± 0.19|

## Acknowledgment


