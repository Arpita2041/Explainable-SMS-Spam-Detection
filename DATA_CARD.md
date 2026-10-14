# Data card

Dataset: UCI SMS Spam Collection. Source format is label-tab-text, UTF-8.

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Attribution: Almeida, T., and Hidalgo, J. (2011). SMS Spam Collection.
UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CC84

UCI lists the dataset under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
The original corpus is downloaded on the user's machine and excluded from Git.
The downloader records the actual source-file SHA-256.

The cleaning audit records original rows, label conflicts, removed duplicates,
and rows used. Labels encode spam as one and ham as zero. Exact normalized-text
duplicates cannot cross the train, validation, and test splits. No sender
identifiers or timestamps support sender-independent or temporal evaluation.

data/demo/sms_demo.tsv consists of original, simulated messages for software
demonstration. The templates make the task deliberately easy; its scores are
not UCI results. These original fixtures use the repository's MIT license.
