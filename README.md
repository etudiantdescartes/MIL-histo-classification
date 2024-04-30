# Multiple Instance Learning based whole slide image classification
WSIs are too large to be processed in one go by a neural network, so the idea is to divide them into tiles:
<div style="text-align: center;">
  <img src="grid.png?raw=true" alt="grid" style="width: 500px; height: 500px; display: inline-block;">
</div>
We then pass each tile through a feature extractor. Here it is trained in self supervision on histological images.
The features vectors are then encoded as nodes and linked together when the nodes represent adjacent tiles in the original image.
An aggregation/classification model is trained to classify these graphs, providing a label at the WSI level.
The training time is actually quite short since the graphs are small. The preprocessing is the longest part.


# Files overview
- ```openslide_extraction.py``` Tissue segmentation and patch extraction WSIs
- ```visualization.py``` Visualization of graphs created from WSIs and attention maps for GAT layers
- ```train.py``` Training and evaluation
- ```model.py``` Classification models
- ```feature_extraction.py``` Patch feature extraction
- ```dataset_creation.py``` Creating graphs using patch coordinates (8-connected adjacent patches are linked in the graph)


