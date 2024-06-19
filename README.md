# Multiple Instance Learning based whole slide image classification
The goal of this project consists in implementing a data processing pipeline, specifically for large histological image (whole slide images or WSIs) binary classification (tumoral or non-tumoral tissue). In the readme section, you'll find the motivation for this project and the result obtained using the Camelyon16 dataset.

# Tile extraction
The main issue in this task is the size of each WSI, having billions of pixels. To solve this, the first step is to divide each WSI into smaller sized tiles. For this purpose, the script ```openslide_extraction.py``` allows you to process each WSI by first breaking it down into 224*224 tiles (this size was chosen mostly for practical purposes regarding the next step). The script then converts each RGB tile to HSV, then run a thresholding on the saturation channel to keep only the tiles containing biological tissue.

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


https://drive.google.com/file/d/1YUafo9802L_GRpN_jOS1neH3aALlcwYv/view?usp=drive_link
