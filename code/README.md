# Code
Here are the scripts used in my thesis

## Modules

* customTree.py
    * The implementation of custom class which read the Imagenet label tree and calculate the label similarity

* customFolder.py
    * The implementation of my custom dataloader in thesis

* Calculate_similarity.ipynb
    * The file calculate label similarity and save

* Outcome.ipynb
    * The file to visualize outcome

## Experiments

* resnet18_original.py
    * Train the model with traditional cross-entropy loss

* resnet18_original_cusSet.py
    * Train the model with traditional cross-entropy loss and custom dataset in my thesis

* resnet18_cosLoss.py
    * Train the model with cross-entropy with deep feature loss in my thesis

* resnet18_cosLoss_cusSet.py
    * Train the model with cross-entropy with deep feature loss and custom dataset in my thesis

* resnet18_simLoss.py
    * Train the model with similarity cross-entropy loss in my thesis

* resnet18_simcosLoss_cusSet.py
    * Train the model with similarity cross-entropy with deep feature loss and custom dataset in my thesis

* resnet18pretrained_continue.py
    * Continue to train the pretrained model with cross-entropy loss, to campare the result with other loss

* resnet18pretrained_coslossCusSet.py
    * Continue to train the pretrained model with cross-entropy with deep feature feature loss and custom dataset in my thesis

* resnet18_simLoss.py
    * Continue to train the pretrained model with similarity cross-entropy loss and custom dataset in my thesis