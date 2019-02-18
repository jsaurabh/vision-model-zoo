This is an implementation of a ResNet 34 on the Stanford [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset

I ended up using only the train data from the dataset, as my model is only learning to do object recognition rather than boundary box detection which is what the dataset is aimed for

The data was obtained from the link given above, and the model was trained using the fastai lib. Steps for preprocessing the data and a general write up will follow soon on my blog.

__3D Object Representations for Fine-Grained Categorization__

Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei

4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.
[pdf](https://ai.stanford.edu/~jkrause/papers/3drr13.pdf)   [BibTex](https://ai.stanford.edu/~jkrause/papers/3drr13.bib)   [slides](https://ai.stanford.edu/~jkrause/papers/3drr_talk.pdf)