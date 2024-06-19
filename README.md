# Multiple Instance Learning based whole slide image classification
The goal of this project consists in implementing a data processing pipeline, specifically for large histological image (whole slide images or WSIs) binary classification (tumoral or non-tumoral tissue). In the readme section, you'll find the motivation for this project and the result obtained using the Camelyon16 dataset.

# Tile extraction
The main issue in this task is the size of each WSI, having billions of pixels. To solve this, the first step is to divide each WSI into smaller sized tiles. For this purpose, the script ```openslide_extraction.py``` allows you to process each WSI by first breaking it down into 224*224 tiles (this size was chosen mostly for practical purposes regarding the next step). The script then converts each RGB tile to HSV, then run a thresholding on the saturation channel to keep only the tiles containing biological tissue. Here is result on one WSI:
<div style="text-align: center;">
  <img src="grid.png?raw=true" alt="grid" style="width: 500px; height: 500px; display: inline-block;">
</div>

# Feature extraction
The next step is feature extraction. Now that we transformed each WSI into sets of tiles, we still have too much data to train a classifier. For this purpose, we first have to extract the features of each tile using a CNN. In my case, I decided to use a pretrained resnet (https://github.com/ozanciga/self-supervised-histopathology), that was trained on 400 000 tiles in a contrastive self-supervised manner. It is easier to use a pretrained model, specifically a CNN, considering my hardware resources. The ```feature_extraction.py``` script can be used to extract the features of each tile, and save them as .npz files.

# Data formatting
Now that the features are extracted, the dataset is much easier to process. The initial goal of the project was to classify WSI, so we now have to arrange the features as "bags", so that a model can be trained to aggregate the features within each bag to eventually provide a label for the corresponding WSIs. Each bag contains the features of every tile from a single WSI. I chose to use a graph neural network for the aggregation / classification phase. The ```dataset_creation.py``` script can be used to create graphs for each bag. Basically, each tile (or feature vector) is represented by a node, and tile adjacency is represented by an edge, in an 8-connected fashion. The grid-like graphs are then saved as .pt files.

# Aggregation / classification
As it was said earlier, the aggregation model is a GNN. It essentially consists in a GATConv layer that allows each node to aggregate neighboring information using an attention mechanism, an attention-based pooling layer to obtain a graph representation in the form of a vector of length 512, and two dense layers.

# Training / evaluation
396 WSIs from the 400 in the Camelyon16 dataset were used to train the model. The hyperparameters were chosen empirically:
- Learning rate: 0.00005
- Weight decay: 5e-3
- Early stopping patience: 10
- Positive class weight: ~1.4
- Batch size: 4
- Epochs: 50
The ```train.py``` script can be used to train and evaluate the model, using a 5-fold cross validation strategy. The metric used for the evaluation is AUROC.
Here is the evaluation result after training the model:
<div style="text-align: center;">
  <img src="auroc.png?raw=true" alt="grid" style="width: 500px; height: 500px; display: inline-block;">
</div>
As we can see, the result is quite high, without reaching SOTA performance. Each fold seems to yield approximately the same result.
Some improvements could be made regarding the evaluation: given more time and resources, it would be interesting to test the model on data from a different source. It would also be interesting to conduct an in-depth study on the regions that are attended by the model, as well as their impact on the classification score. Another possibility would be trying other feature extractors or implementing a new one.
