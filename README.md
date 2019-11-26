# Character-level Convolutional Neural Network for Hate Speech Detection

As of January 2019 there were 3.48 billion global social media users per month. This has unfortunately contributed 
to a rise in the rate of hate speech distributed online. Despite attempts by platforms such as Facebook, Twitter, 
Reddit, and increasing pressure from Governments, the problem of detection and removal of hate speech has proven 
difficult to address.
	    
In the past decade, various state-of-the-art detection models have been proposed, including a cutting-edge industry 
solution - Google Perspective. Nevertheless, recent research has shown these detection models are fragile when 
adversarial approaches are introduced to intentionally avoid detection. Therefore this paper will demonstrate the 
capability of Character-level Convolutional Neural Networks (CNN) for hate speech detection. The CNN will be shown 
to retain a high level of detection accuracy, even for adversarial content, in part due to improved adversarial 
training.

## How to use the code

Launch predict.py with the following arguments:

- `-m --model`: Choose a model to make predictions: 'cnn+', 'cnn', 'logreg', 'svm'.
- `-s --string`: Provide a single string for classification
- `-f -- filepath`: Provide a file containing tweets for classification
- `-w --write`: Define a file path to write predictions to

Example usage:

```bash
python predict.py -s "I hate handicap f*ggots" -m cnn+ -w data/private

```

## Dependencies

- numpy 
- pandas
- sklearn
- Keras
- Tensorflow
