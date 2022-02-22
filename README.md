# EASY - Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients.
This repository is the official implementation of [EASY - Ensemble Augmented-Shot Y-shaped Learning: State-Of-The-Art Few-Shot Classification with Simple Ingredients](https://arxiv.org/pdf/2201.09699.pdf).

EASY proposes a simple methodology, that reaches or even beats state of the art performance on multiple standardized benchmarks of the field, while adding almost no hyperparameters or parameters to those used for training the initial deep learning models on the generic dataset.

## Downloads 
Please click the [Google Drive link](https://drive.google.com/drive/folders/1fMeapvuR6Rby0HDHd5L74BEXRyiOF942) for downloading the features, backbones and datasets.

Each of the files (backbones and features) have the following prefixes depending on the backbone: 

|  Backbone  | prefix | Number of parameters |  
|:--------:|:------------:|:------------:|
| ResNet12 | | 12M|
| ResNet12(1/sqrt(2)) | small | 6M|
| ResNet12(1/2) | tiny | 3M|

Each of the features file is named as follow : 
- if not AS : "<backbone_prefix><dataset_name>features<backbone_number>.pt<backbone_suffix>"
- if AS     : "<backbone_prefix><dataset_name>featuresAS<backbone_number>.pt<backbone_suffix>"

Suffixes <backbone_suffix>: 
- .pt11 : For 1-shot classification, the best backbone selected during training is based on the 1-shot performance of the validation dataset.
- .pt55 : For 5-shot classification, the best backbone selected during training is based on the 5-shot performance of the validation dataset.

## Testing scripts for EASY
Run scripts to evaluate the features on FSL tasks for Y and ASY. For EY and EASY use the corresponding features in 1-shot setting. For 5-shot setting, change --n-shots to 5

### Inductive setup using NCM
Test features on miniimagenet using Y (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11' --preprocessing ME --n-shots 1

Test features on miniimagenet using ASY (Resnet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeaturesAS1.pt11' --preprocessing ME --n-shots 1

Test features on miniimagenet using EY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeatures1.pt11, <path>/minifeatures2.pt11, <path>/minifeatures3.pt11]" --preprocessing ME --n-shots 1
    
Test features on miniimagenet using EASY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeaturesAS1.pt11, <path>/minifeaturesAS2.pt11, <path>/minifeaturesAS3.pt11]" --preprocessing ME --n-shots 1


### Transductive setup using Soft k-means
Test features on miniimagenet using Y (ResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeatures1.pt11'--postprocessing ME --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

Test features on miniimagenet using ASY (ResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features '<path>/minifeaturesAS1.pt11' --postprocessing ME --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

Test features on miniimagenet using EY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeatures1.pt11, <path>/minifeatures2.pt11, <path>/minifeatures3.pt11]" --postrocessing ME  --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

Test features on miniimagenet using EASY (3xResNet12)

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --test-features "[<path>/minifeaturesAS1.pt11, <path>/minifeaturesAS2.pt11, <path>/minifeaturesAS3.pt11]" --postrocessing ME  --transductive --transductive-softkmeans --transductive-temperature-softkmeans 5 --n-shots 1

## Training scripts for Y
Train a model on miniimagenet using manifold mixup, self-supervision and cosine scheduler. The best backbone is based on the 1-shot performance in the validation set. In order to get the best 5-shot performing model during validation, change --n-shots to 5 :

    $ python main.py --dataset-path "<dataset-path>" --dataset miniimagenet --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing ME --n-shots 1 --skip-epochs 450 --save-model "<path>/mini<backbone_number>.pt1"

## Important Arguments
Some important arguments for our code.

**Training arguments**
- `dataset`: choices=['miniimagenet', 'cubfs','tieredimagenet', 'fc100', 'cifarfs']
- `model`: choices=['resnet12', 'resnet18', 'resnet20', 'wideresnet', 's2m2r']
- `dataset-path`: path of the datasets folder which contains folders of all the datasets.
- `rotations` : if mentionned, self-supervision will be used during training.
- `cosine` : if mentionned, cosine scheduler will be used during training.
- `save-model`: path where to save the best model based on validation data.
- `manifold-mixup`: number of epochs where to use manifold-mixup.
- `skip-epochs`: number of epochs to skip before evaluating few-shot performance. Used to speed-up training.
- `n-shots` : how many shots per few-shot run, can be int or list of ints. 

**Few-shot Classification**
- `preprocessing`: preprocessing sequence for few shot given as a string, can contain R:relu P:sqrt E:sphering and M:centering using the base data.
- `postprocessing`: postprocessing sequence for few shot given as a string, can contain R:relu P:sqrt E:sphering and M:centering on the few-shot data, used for transductive setting.

## Few-shot classification Results

Experimental results on few-shot learning datasets with ResNet-12 backbone. We report our average results with 10000 randomly sampled episodes for both 1-shot and 5-shot evaluations.

**MiniImageNet Dataset (inductive)**

| Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
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
|EASY 2×ResNet12 1/√2  (ours) |70.63 ± 0.20| 86.28 ± 0.12|
|above <=12M|***nb of parameters***|below 36M|
|3S2M2R [12] | 64.93 ± 0.18 | 83.18 ± 0.11 |
|LR + DC [17] | 68.55 ± 0.55 | 82.88 ± 0.42 |
|EASY 3×ResNet12 (ours) | 71.75 ± 0.19 | 87.15 ± 0.12 |

**TieredImageNet Dataset (inductive)**

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
|above <=12M|***nb of parameters***|below 36M|
|S2M2R [12] |73.71 ± 0.22| 88.52 ± 0.14|
|EASY 3×ResNet12 (ours) |74.71 ± 0.22| 88.33 ± 0.14|


**CUBFS Dataset (inductive)**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|FEAT [36]| 68.87 ± 0.22| 82.90 ± 0.10|
|LaplacianShot [41]| 80.96 |88.68|
|ProtoNet [10] |66.09 ± 0.92 |82.50 ± 0.58|
|DeepEMD v2 [13] |79.27 ± 0.29| 89.80 ± 0.51|
|EASY 4×ResNet12 1/sqrt(2) |77.97 ± 0.20 | 91.59 ± 0.10|
|above <=12M|***nb of parameters***|below 36M|
|S2M2R [12] | 80.68 ± 0.81 | 90.85 ± 0.44|
|EASY 3×ResNet12 (ours) | 78.56 ± 0.19 | 91.93 ± 0.10 |


**CIFAR-FS Dataset (inductive)**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|S2M2R [12] |63.66 ± 0.17 | 76.07 ± 0.19|
|R2-D2 (+ens) [20] | 76.51 ± 0.47 | 87.63 ± 0.34|
|invariance-equivariance [39] | 77.87 ± 0.85 | 89.74 ± 0.57|
|EASY 2×ResNet12 1/sqrt(2) (ours) | 75.24 ± 0.20 | 88.38 ± 0.14|
|above <=12M|***nb of parameters***|below 36M|
|S2M2R [12] |74.81 ± 0.19| 87.47 ± 0.13|
|EASY 3×ResNet12 (ours) | 76.20 ± 0.20 | 89.00 ± 0.14|

**FC-100 Dataset (inductive)**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|DeepEMD v2 [13] | 46.60 ± 0.26 | 63.22 ± 0.71|
|TADAM [35] | 40.10 ± 0.40 | 56.10 ± 0.40|
|ProtoNet [10] | 41.54 ± 0.76 | 57.08 ± 0.76|
|invariance-equivariance [39] |47.76 ± 0.77 | 65.30 ± 0.76|
|R2-D2 (+ens) [20] | 44.75 ± 0.43 | 59.94 ± 0.41|
|EASY 2×ResNet12 1/sqrt(2) (ours)| 47.94 ± 0.19 | 64.14 ± 0.19|
|above <=12M|***nb of parameters***|below 36M|
|EASY 3×ResNet12 (ours) | 48.07 ± 0.19 | 64.74 ± 0.19|


**Minimagenet (transductive)**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|TIM-GD [42] |73.90| 85.00|
|ODC [43] |77.20 ± 0.36 |87.11 ± 0.42|
|PEMnE-BMS∗ [32] |80.56 ± 0.27| 87.98 ± 0.14|
|SSR [44] |68.10 ± 0.60| 76.90 ± 0.40|
|iLPC [45] |69.79 ± 0.99| 79.82 ± 0.55|
|EPNet [31]| 66.50 ± 0.89 |81.60 ± 0.60|
|DPGN [46] |67.77 ± 0.32| 84.60 ± 0.43|
|ECKPN [47]| 70.48 ± 0.38| 85.42 ± 0.46|
|Rot+KD+POODLE [48]| 77.56| 85.81|
|EASY 2×ResNet12( 1√2) (ours) |81.70 ±0.25 |88.29 ±0.13|
|above <=12M|***nb of parameters***|below 36M|
|SSR [44] |72.40 ± 0.60 |80.20 ± 0.40|
|fine-tuning(train+val) [49]| 68.11 ± 0.69| 80.36 ± 0.50|
|SIB+E3BM [50] |71.40 |81.20|
|LR+DC [17] |68.57 ± 0.55| 82.88 ± 0.42|
|EPNet [31] |70.74 ± 0.85 |84.34 ± 0.53|
|TIM-GD [42] |77.80 |87.40|
|PT+MAP [51] |82.92 ± 0.26 |88.82 ± 0.13|
|iLPC [45] |83.05 ± 0.79| 88.82 ± 0.42|
|ODC [43] |80.64 ± 0.34| 89.39 ± 0.39|
|PEMnE-BMS∗ [32] |83.35 ± 0.25| 89.53 ± 0.13|
|EASY 3×ResNet12 (ours) |84.04 ±0.23 |89.14 ±0.11|

**CUB-FS (transductive)**
|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|TIM-GD [42] |82.20| 90.80|
|ODC [43]| 85.87| 94.97|
|DPGN [46] |75.71 ± 0.47| 91.48 ± 0.33|
|ECKPN [47]| 77.43 ± 0.54 |92.21 ± 0.41|
|iLPC [45]| 89.00 ± 0.70| 92.74 ± 0.35|
|Rot+KD+POODLE [48] |89.93| 93.78|
|EASY 4×ResNet12( 1/2) (ours) |90.41 ± 0.19| 93.58 ± 0.10|
|above <=12M|***nb of parameters***|below 36M|
|LR+DC [17] |79.56 ± 0.87| 90.67 ± 0.35|
|PT+MAP [51]| 91.55 ± 0.19| 93.99 ± 0.10|
|iLPC [45] |91.03 ± 0.63| 94.11 ± 0.30|
|EASY 3×ResNet12 (ours)| 90.56 ± 0.19| 93.79 ± 0.10|


**CIFAR-FS (transductive)**

|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|SSR [44] |76.80 ± 0.60 |83.70 ± 0.40|
|iLPC [45] |77.14 ± 0.95| 85.23 ± 0.55|
|DPGN [46] |77.90 ± 0.50| 90.02 ± 0.40|
|ECKPN [47] |79.20 ± 0.40| 91.00 ± 0.50|
|EASY 2×ResNet12 (1/sqrt(2)) (ours) |86.40 ± 0.23 |89.75 ± 0.15|
|above <=12M|***nb of parameters***|below 36M|
|SSR [44] |81.60 ± 0.60| 86.00 ± 0.40|
|fine-tuning (train+val) [49] |78.36 ± 0.70 |87.54 ± 0.49|
|iLPC [45]| 86.51 ± 0.75| 90.60 ± 0.48|
|PT+MAP [51] |87.69 ± 0.23| 90.68 ± 0.15|
|EASY 3×ResNet12 (ours)| 87.16 ± 0.21| 90.47 ± 0.15|

**FC-100 (transductive)**
|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|EASY 2×ResNet12( 1√2)(ours) |54.68 ± 0.25 |66.19 ± 0.20|
|above <=12M|***nb of parameters***|below 36M|
|SIB+E3BM [50] |46.00 |57.10|
|fine-tuning (train) [49] |43.16 ± 0.59| 57.57 ± 0.55|
|ODC [43]| 47.18 ± 0.30| 59.21 ± 0.56|
|fine-tuning (train+val) [49]| 50.44 ± 0.68 |65.74 ± 0.60|
|EASY 3×ResNet12 (ours)| 54.13 ± 0.24 |66.86 ± 0.19|

**Tiered Imagenet (transducive)**
|  Methods  | 1-Shot 5-Way | 5-Shot 5-Way |   
|:--------:|:------------:|:------------:|
|PT+MAP [51] |85.67 ± 0.26| 90.45 ± 0.14|
|TIM-GD [42] |79.90| 88.50|
|ODC [43]| 83.73 ± 0.36 |90.46 ± 0.46|
|SSR [44] |81.20 ± 0.60| 85.70 ± 0.40|
|Rot+KD+POODLE [48] |79.67 |86.96|
|DPGN [46] |72.45 ± 0.51 |87.24 ± 0.39|
|EPNet [31] |76.53 ± 0.87 |87.32 ± 0.64|
|ECKPN [47] |73.59 ± 0.45 |88.13 ± 0.28|
|iLPC [45] |83.49 ± 0.88| 89.48 ± 0.47|
|ASY ResNet12 (ours)| 82.66 ± 0.27 |88.60 ± 0.14|
|above <=12M|***nb of parameters***|below 36M|
|SIB+E3BM [50] |75.60 |84.30|
|SSR [44] |79.50 ± 0.60| 84.80 ± 0.40|
|fine-tuning (train+val) [49] |72.87 ± 0.71| 86.15 ± 0.50|
|TIM-GD [42]| 82.10 |89.80|
|LR+DC [17] |78.19 ± 0.25 |89.90 ± 0.41|
|EPNet [31] |78.50 ± 0.91 |88.36 ± 0.57|
|ODC [43] |85.22 ± 0.34| 91.35 ± 0.42|
|iLPC [45] |88.50 ± 0.75| 92.46 ± 0.42|
|PEMnE-BMS∗ [32]| 86.07 ± 0.25 |91.09 ± 0.14|
|EASY 3×ResNet12 (ours)| 84.29 ± 0.24| 89.76 ± 0.14|
