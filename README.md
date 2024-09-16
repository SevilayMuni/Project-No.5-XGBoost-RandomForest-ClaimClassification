
# XGBoost Predictive Model for TikTok's Claim Classification: EDA, Hypothesis Testing, Logistic Regression, Tree-Based Models

The TikTok team wants to develop a machine learning model to classify claims for user submissions. 
I carried out **EDA** on the dataset. The goal of EDA was to investigate the impact that videos have on TikTok users.
**Statistical analysis** and **hypothesis testing** verified that if a user is verified, they are much more likely to post opinions. Thus, a **logistic regression** model that predicts verified_status was built.

I built **random forest** and **XGBoost** models. Both models were used to predict on validation dataset, and final model selection was determined by the model with the best recall score. **XGBoost was chosen as a final model, then it is used to score a test dataset to estimate future performance.**
*The confusion matrix above displays the performance of final XGBoost model on the test data.*
[<img src="https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-Models/Confusion-Matrix-XGBoost-TestData.png" width="700"/>](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-Models/Confusion-Matrix-XGBoost-TestData.png)

## Business Understanding
TikTok videos receive a large number of user reports for many different reasons. Not all reported videos can be reviewed by a human moderator. Videos that make claims are much more likely to contain content that violates the TikTok’s terms of service. **The goal is developing method to identify videos that make claims to prioritize them for review.** 

## Data Understanding
- Of total 19,382 samples in this dataset, just under 50% are claims — 9,608.
- **Engagement level is strongly correlated with claim status.**

[<img src="https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-EDA/Author-Ban-Status-vs-Avg-View-Count.png" width="800"/>](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-EDA/Author-Ban-Status-vs-Avg-View-Count.png)
- Banned authors and those under review get more views, likes, and shares than active authors.
- **Banned authors have a median share count 33 times the median share count of active authors!**

![image1](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-EDA/Distribution-of-Total-Views-vs-Claim-Status.png)
- **Claim videos have a higher view rate** than opinion videos.

![image2](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-EDA/Video-View-Count-vs-Like-Count-Claim-Status.png)
- **Claim videos also have a higher rate of likes** on average, so they are more favorably received as well.
- **Claim videos receive more engagement via comments and shares** than opinion videos.
- For claim videos, banned authors have slightly higher likes/view and shares/view than active authors or those under review.
- For opinion videos, active authors and those under review both get higher engagement rates than banned authors in all categories.

![image3](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-EDA/Claim-Verification-Status-Histogram.png)
- There are noticeably **fewer verified users than unverified** users.
- **Verified users are much more likely to post opinions.**


      TtestResult(statistic=25.499441780633777, pvalue=2.6088823687177823e-120, df=1571.163074387424)

**There is statistically significant difference** in views between videos posted by verified accounts and videos posted by unverified accounts. It indicates **key behavioral differences between these account groups**.

The ‘verified_status’ variable was analyzed in this regression model since the relationship between account type and the video content was indicated previously. **A logistic regression model was selected because of the data type and distribution.**
The logistic regression model yielded: 
- precision of 67% 
- recall of 65% 
- F1 score of 63%
**The logistic regression model had decent predictive power.** Based on the estimated model coefficients from the logistic regression, **longer videos tend to be associated with higher odds of the user being verified**.

## Modeling and Evaluation
### Model Design and Target Variable
Column 'claim_status' is binary value that indicates whether a video is a claim or an opinion. It will be the target variable.
This is a **classification task since the model is predicting a binary class**.

### Selecting an Evaluation Metric
To determine which evaluation metric might be best, how the model might can be wrong should be considered.
There are two possibilities for bad predictions:
- False positives (FP): When the model predicts a video is a claim when in fact it is an opinion
- False negatives (FN): When the model predicts a video is an opinion when in fact it is a claim
**It's significant to identify videos that violate the terms of service, even if that means some opinion videos are misclassified as claims.**

**Because it's more important to minimize FNs, the model evaluation metric will be recall.**

### Modeling Workflow & Model Selection Process
Previous analysis of the dataset has showed that there are ~20,000 videos in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:
1) Split the data into train/validation/test sets (60/20/20)
2) Fit models and tune hyperparameters on the training set
3) Perform final model selection on the validation set
4) Assess the champion model's performance on the test set

### Tuning Hyperparameters
Parameters used for **random forest**:
{'max_depth': 5,
 'max_features': 0.5,
 'max_samples': 0.8,
 'min_samples_leaf': 2,
 'min_samples_split': 4,
 'n_estimators': 75}

Parameters used for **XGBoost**:
{'learning_rate' = 0.1, 
'max_depth' = 4, 
'min_child_weight' = 5, 
'n_estimators' = 300}


## Model Results

    Random Forest   Precision  Recall   F1  
    weighted avg.   0.99       0.99     0.99
    
    XGBoost   Precision  Recall   F1  
    weighted avg.   1.00       1.00     1.00
## Conclusion

Both random forest and XGBoost models performed exceptionally well. 
**The XGBoost had a better precision and recall score and was selected as the champion.**
- Performance on the test holdout data yielded near perfect scores, with only 21 misclassified samples out of 3,817.

Subsequent analysis indicated that the primary predictor was ‘video view count’ -- related to video engagement levels. 
Overall, **videos with higher user engagement levels were much more likely to be claims.**
- In fact, no opinion video had more than 10,000 views.

![image4](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-Models/Feature-Importance-XGBoost.png)
For **XGBoost**, the most predictive features:
- 'video_view_count' 
- 'video_share_count'
- 'author_ban_status_banned'

![image5](https://github.com/SevilayMuni/Project-No.5-XGBoost-RandomForest-ClaimClassification/blob/main/graphs-from-Models/Feature-Importance-Random-Forest.png)
For **Random Forest**, the most predictive features:
- 'video_view_count'
- 'video_like_count'
- 'video_share_count'
- 'video_comment_count'

*Random Forest model yielded more features compare to XGBoost.* **!** Since both model performed very similarly on validation dataset, one can also choose Random Forest as its champion model **!**

### Next Steps
Before deploying the model, further evaluation using additional subsets of user data is recommended. In addition, monitoring the distributions of video engagement levels to ensure that the model remains robust to fluctuations in its most predictive features is suggested. 



## Acknowledgements
Data Source: Google Advanced Data Analytics Course

Templates for executive summary: Google Advanced Data Analytics Course

Readme.so
