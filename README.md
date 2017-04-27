## UPDATE 2017.04.27:
### 1. updated example.py for using word and/or lemma n-grams as base feature:
use only word n-grams: 

```
get_base_feature(target = ['word'])
```

use both word n-grams and lemma n-grams:

```
get_base_feature(target = ['word','lemma'])
```


### 2. Including other features such as character n-grams returned by function```get_char_ngram()``` by using:

```
hstack([train_X_base, train_X_char], format='csr')
```

and

```
hstack([dev_X_base,dev_X_char], format='csr')
```

other feautures can be included in a similar way.

#### some experiment results:

1. The result obtained on 2017 dataset is slightly worse than 2013 dataset, which could be because that they modified some data in 2017 dataset.

2. So far I've tried:

   

Feature | weighting | Dev acc.
---|---|---
word + lemma n-grams (1,2,3) | logent | 84.09
word + lemma n-grams (1,2,3) (including punctuation)| logent | 84.82
word + lemma + char n-grams (1,2,3) | logent |84.54
word + lemma + char n-grams (1,2,3) | bin | 80.45