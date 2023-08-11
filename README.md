# Comment Toxicity Predictor:

## 1. Business Problem:
**Source:** https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

**Description:** https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/description

**Problem Statement:** Given a comment made by the user, predict the toxicity of the comment.


## 2. Machine Learning Problem Formulation:

### 2.1 Data-Description: 

- Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
- We have one single csv file for training and one cvs file to test.
- Columns in train data:
	- Comment_text: This is the data in string format which we have to use to find the toxicity.
	- target: Target values which are to be predicted (has values between 0 and 1)
	- Data also has additional toxicity subtype attributes: (Model does not have to predict these)
		- severe_toxicity
		- obscene
		- threat
		- insult
		- identity_attack
		- sexual_explicit
	
### 2.2 Example Datapoints and Labels:

**Comment:** i'm a white woman in my late 60's and believe me, they are not too crazy about me either!!

- Toxicity Labels: All 0.0

### 2.3 Type of Machine Learning Problem:
We have to predict the toxicity level(target attribute). The values range from 0 to 1 inclusive. This is a regression problem. It can also be treated as a classification problem if we take every value below 0.5 to be non-toxic and above it to be toxic, we would then get a binary classification problem.

## 3 Model Description
### 3.1 Methodology:
The toxicity prediction model is based on a deep learning architecture using LSTM which takes vector of each comment as input and predicts for all the different labels as seperate binary using sigmoid activation. Further we extended it by integrating it with gradio web app via which we can take input and predict in real time. The specifics of the model architecture and training can be found in the Toxicity directory.
### 3.2 Tech stack:
- Python
- TensorFlow
- Natural Language Processing
- Gated Reccurent Units (LSTM, Bidirectional)
- Gradio Web app
