# SENTIMENT ANALYSIS & POLARITY SHIFTING
The project is inspired by [Shoushan Li, et al. 2010](http://dl.acm.org/citation.cfm?id=1873853), which provided the framework for Sentiment Analysis with Polarity Shifting.  
Dataset collected by [Blitzer et al. 2007](http://www.seas.upenn.edu/~mdredze/datasets/sentiment/).  
  

### How to run

* See config.py to setup resources folder & other parameter  
	- LAMDA: LAMDA to calculate WFO. default = 0
	- N_MAX: Limit the number of sentences of each type use to train polarity shifting detection model. defalt = 1000 (=2000 shift + 2000 unshift sentences)
	- N_GRAM: Unigram or Bigram
	- MODEL: SVM or Logistic Regression
  
* Preprocess raw data:   
	>>python handle_unprocessed_data.py [domain]  
  
* Calculate WFO and create ranking words:  
	python generate_shifting_training_data.py [domain]  
  
* Train the polarity shifting detection model:  
	>>python tfidf_model.py -detect [domain]
	python shifting_detection_model.py [domain]  
  
* Train the original baseline model:
	>>python tfidf_model.py -base [domain]  
 	python baseline_model.py [domain]    
  
* Train the shifted baseline model:  
	>>python tfidf_shift_model.py [domain]  
	python shifted_model.py [domain]  

* Train the unshifted baseline model:  
	>>python ensemble_model.py [domain]  

* Note: domain = ['books', 'dvd', 'electronics', 'kitchen']  | -all

### Result:  

Result in Unigram, Logistic Regression  
```
|                            | Books | DVD   | Electronics | Kitchen |  
|----------------------------|-------|-------|-------------|---------|  
| Original Baseline          | 79.80 | 81.10 | 81.90       | 85.00   |  
| Shifted Baseline           | 71.50 | 70.50 | 70.90       | 73.37   |  
| Unshifted Baseline         | 75.30 | 76.50 | 75.10       | 78.50   |  
| Ensemble - Product Rule    | 80.90 | 81.90 | 82.30       | 85.60   |  
| Ensemble - Stacking Method | 81.10 | 81.90 | 82.70       | 86.10   |  
```
P value and statistical significance:  
* The two-tailed P value equals 0.0038  
* This difference is considered to be very statistically significant  

-------
##### REFERENCE
Li, Shoushan, et al. "Sentiment classification and polarity shifting." Proceedings of the 23rd International Conference on Computational Linguistics. Association for Computational Linguistics, 2010.
