# Aligned SOM

Aligned SOMs aims at training mulitple layers of n SOMs with differently weighted subsets of attributes.
The implementation of the SOM training is closely modelled after the decription in the paper Aligned self-organizing maps by Pampalk, Elias [[1]](#1).

The Alignd SOM implementation uses the well known MiniSom package and trains multiple layers of the MiniSom [[2]](#2). A Layer has in extension to the normal MiniSom implementation the possibility to set initial codebook weights. Furter, the update method is adapted to model the distance between layers. We implemented an online-training algorithm which iteratively traines all layers.


## Setup
* todo: add setup description
* todo: add current requirements.txt freez

## References
<a id="1">[1]</a>
Pampalk, Elias.
"Aligned self-organizing maps." Proceedings of the Workshop on Self-Organizing Maps. 2003.   
URL: https://www.researchgate.net/publication/2887633_Aligned_Self-Organizing_Maps

<a id="2">[2]</a>
Vettigli, Giuseppe.
"MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map." (2018).   
URL: https://github.com/JustGlowing/minisom 