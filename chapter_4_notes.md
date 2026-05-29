# Chapter 4 - Classification and Regression

- Definitions:
    - Sample/Input: One data point that goes into the model
    - Prediction/Output: What comes out of the model.
    - Target: Truth. What the model should ideally have predicted, according to an external source of data.
    - Prediction Error/Loss Value: A measure of the distance between your model’s prediction and the target.
    - Classes: A set of possible labels to choose from in a classification problem. 
        - classifying cat and dog pictures: `“dog”` and `“cat”` are the two classes.
    - Labels: A specific instance of a class annotation in a classification problem.
    - Ground-truth or annotations: All targets for a dataset, typically collected by humans.
    - Binary Classification: A classification task where each input sample should be categorized into two exclusive categories.
    - Categorical Classification or multiclass classification: A classification task where each input sample should be categorized into more than two categories: for instance, classifying handwritten digits.
    - Multilabel Classification: A classification task where each input sample can be assigned multiple labels. For instance, a given image may contain both a cat and a dog and should be annotated with both the “cat” label and the “dog” label. The number of labels per image is usually variable.
    - Scalar Regression: A task where the target is a continuous scalar value. Predicting house prices is a good example: the different target prices form a continuous space.
    - Vector Regression: A task where the target is a set of continuous values: for example, a continuous vector. If you’re doing regression against multiple values (such as the coordinates of a bounding box in an image), then you’re doing vector regression.
    - Mini-batch or just batch: A small set of samples (typically between 8 and 128) that are processed simultaneously by the model. 
        - often a power of 2, to facilitate memory allocation on GPU. 
        - When training, a mini-batch is used to compute a single gradient-descent update applied to the weights of the model.
