# ID3
## Purpose
The objective of the program is to classify different examples using the ID3 algorithm.

## Rules
- To train and validate the data is used 10-fold cross-validation.
- The data is randomly separated in 10 sets on each program start.
- Use different pruning methods to avoid overfitting.

## About the program
The program is using the **ID3** algorithm to classify different examples in two classes - recurrence events and no recurrence events. Also it compares two ways of pre-pruning - based on examples left, in this case 15, and based on max depth, in this case 2. The program displays results when no pruning is done as well.

- Used data - [Here](https://archive.ics.uci.edu/ml/datasets/breast+cancer)

### To run the program
- Run `python3 main.py`

### Output
- Outputs accuracy for each of the 10 trainings and the average accuracy for a summary assessment of the classifier.
- The information is displayed to compare classification with no pruning, pre-pruning based on examples and pre-pruning based on depth.

### Example
![Example](https://github.com/luntropy/ID3/blob/main/images/output-example.png)
