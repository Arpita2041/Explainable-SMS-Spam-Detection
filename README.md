# Explainable SMS Spam Detection

The main objective is to develop a Machine Learning system that automatically classifies SMS messages as Spam or Ham and explains why a particular classification was made.
	- Spam: Unwanted, suspicious, fraudulent, or unsolicited promotional messages.
	- Ham: Normal, legitimate SMS messages.
Instead of simply predicting Spam or Ham, the system identifies which text features contributed to the prediction using exact feature contributions.

## Setup

The projects require Python 3.11

## Workflow

The project is designed around the UCI SMS Spam Collection Dataset. This experiment used 360 artificially constructed SMS messages. These messages were deliberately designed to test whether the software works.

	- Implemented text preprocessing techniques such as Unicode normalization, case folding, whitespace cleaning, and duplicate removal.
	- Used TF-IDF feature extraction to convert SMS messages into numerical representations.
	- Explored both word-level and character-level TF-IDF features.
	- Compared different Machine Learning algorithms, including Naive Bayes and Logistic Regression, to identify an effective classification model.
	- Implemented a spelling-perturbation test to examine how the models respond to selected changes in spam-related words.

## Result

In the initial software demonstration, the learned models achieved 100% accuracy on a simulated dataset of 360 messages. However, Demonstration model returns ham with spam probability 0.000126 at threshold 0.5. Negative contributions from features such as 'meeting notes', 'notes', 'please', and 'tomorrow' reduce spam log-odds. 

**Development date:** December 2024

~~~bash
python main.py predict --model demo_run/models/model.joblib --text "Please send the meeting notes tomorrow"
~~~
