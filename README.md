# Aligned SOM

Aligned SOMs aims at training mulitple layers of n SOMs with differently weighted subsets of attributes.
The implementation of the SOM training is closely modelled after the decription in the paper Aligned self-organizing maps by Pampalk, Elias [[1]](#1).

The Alignd SOM implementation uses the well known MiniSom package and trains multiple layers of the MiniSom [[2]](#2). A Layer has in extension to the normal MiniSom implementation the possibility to set initial codebook weights. Furter, the update method is adapted to model the distance between layers. We implemented an online-training algorithm which iteratively traines all layers.

The base visualization functions for one Layer and functions for loading the datasets were taken from [PySOMVis](https://github.com/smnishko/PySOMVis).

## Notebooks

We provide three notebooks exploring the Algind SOMs implementation.
* **AlignedSOM_implementation_details:** Description of essential parts of the implemented code in more detail.
* **AlignedSOM_chainlink:** Visual evaluation of the Alignd SOMs on the Chainlink dataset
* **AlignedSOM_10clusters:** Visual evaluation of the Alignd SOMs on the 10Clusters dataset

## Visualization Example on Animals Dataset

We show here a small example on the animals dataset as in the paper "Aligned Self-Organizing Maps" [[1]](#1).

The dataset comprises 16 records of different kinds of animals, described by 13 binary-valued attributes. The animals can be categorised into three classes: birds, carnivores, and herbivores.

The features are split into activity (aspect A) and appearance (aspect B) features.   
**activity features**: hunt, run, fly, swim   
**appearance features**: small, medium, big, 2_legs, 4_legs, hair, hooves, mane, feathers 

![Alt text](animals_dataset_exmple.png?raw=true "Alignd SOM on Animals Dataset")

## How to reproduce

The algorithm and all notebooks were tested with Python 3.9 and all libraries requirements used can be found in the Pipfile or Pipfile.lock.   
The virtual environment can easily be recreated using [Pipenv](https://pipenv.pypa.io/en/latest/index.html) (pipenv install).   
We further provide a freez of the used requirements in the requirements.txt file.

## Data

The datasets were optained from [Data Mining with the Java SOMToolbox](http://www.ifs.tuwien.ac.at/dm/somtoolbox/index.html).   
They can be downloaded directly from the Section [Benchmark data & maps](http://www.ifs.tuwien.ac.at/dm/somtoolbox/datasets.html).

## References

<a id="1">[1]</a>
Pampalk, Elias.
"Aligned self-organizing maps." Proceedings of the Workshop on Self-Organizing Maps. 2003.   
URL: https://www.researchgate.net/publication/2887633_Aligned_Self-Organizing_Maps

<a id="2">[2]</a>
Vettigli, Giuseppe.
"MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map." (2018).   
URL: https://github.com/JustGlowing/minisom 