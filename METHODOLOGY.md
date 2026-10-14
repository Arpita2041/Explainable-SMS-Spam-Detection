# Methodology

Each source line contains a ham or spam label, a tab, and the message.
Text normalization applies Unicode NFKC, case folding, whitespace collapse,
and trimming. It does not remove phone numbers, links, punctuation, or stopwords.

Normalized text is the duplicate key. All rows of a key with contradictory
labels are removed; remaining exact duplicate keys retain their first row.
This occurs before any split. Stratified random indices allocate 60% to training,
20% to validation, and 20% to the final test. Quick mode selects up to 2,200
stratified messages before these splits. Similar templates may still overlap;
the protocol prevents normalized exact-text duplication, not every semantic
near-duplicate or sender overlap.

Word TF-IDF uses unigram and bigram features, minimum document frequency two,
maximum document frequency 0.98, sublinear term frequency, and at most 20,000
features. Character TF-IDF uses word-boundary character n-grams of length three
to five and at most 30,000 features. Vocabulary and IDF are fitted only on
training messages, inside each pipeline.

Word features support Multinomial Naive Bayes and logistic regression.
Character features support logistic regression. A prior-probability baseline is
included. Full and quick modes differ in sample size; they use the same model
settings. This project does not need a downloaded language model or API key.

Model selection maximizes validation average precision. Each candidate's
operating threshold maximizes validation spam F1, with ties resolved toward
0.5. Both choices are frozen before test evaluation. Because the prior baseline's
threshold is also tuned, it can predict spam for every message; it is not
necessarily the majority-class decision rule.

The main metrics are spam precision, recall, F1, macro F1, and average precision.
Accuracy alone can reward a model that misses spam. ROC AUC and log loss are
also recorded. Probabilities are the classifier's outputs; separate probability
calibration is not claimed. Review false positives and false negatives in
error_analysis.csv.

The spelling diagnostic replaces only the keywords free, prize, claim, and
winner with fixed variants. The original labels are retained for this limited
stress test. It does not establish resistance to arbitrary adversarial messages.

For a logistic model, each present feature contributes TF-IDF times its learned
coefficient to the log-odds; the intercept completes the sum. For Naive Bayes,
the coefficients are the differences of class-conditional log probabilities
and the intercept is the log prior odds. The tests verify the reconstructed
log-odds against model probabilities. Positive contributions favor spam.
Only the largest absolute contributions are displayed, while the reported
log-odds use the full feature sum.

The corpus is old and represents a limited collection of sources. Sender
identities and deployment chronology are unavailable. Modern languages,
message styles, and spam prevalence require new validation.
