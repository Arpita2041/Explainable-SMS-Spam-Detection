# Study guide

Read project/data.py before project/models.py. Then follow train and predict
in project/workflow.py.

Be prepared to explain:

1. Why duplicate removal belongs before splitting.
2. Which near-duplicate types the normalization does not remove.
3. How TF, sublinear TF, IDF, and vector normalization affect a message vector.
4. Why the vocabulary and IDF must be fitted on training messages only.
5. The conditional independence assumption in Naive Bayes.
6. What word bigrams and boundary character n-grams capture differently.
7. Why spam recall and legitimate-message false positives both matter.
8. The difference between average precision and thresholded F1.
9. How the selected threshold is saved and used at inference.
10. How the exact feature contribution sum produces log-odds.
11. What an unseen word does to each vectorizer.
12. Why a spelling perturbation score is not a general security guarantee.

For the practical exam, explain a prediction numerically, including its intercept
and omitted small feature contributions. Trace an unseen token through both
vectorizers. Change the word n-gram range or a stated threshold objective,
then justify the comparison using validation data.
