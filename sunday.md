Grant Swigart
6/22/2020

# Introduction

Welcome to our analysis for . During this article we will be analyzing
what makes online news articles popular or unpopular. Luckily all of our
data collection has already been completed. We will be looking a dataset
of [Mashable online news
articles](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).
This dataset has a wide range of valuable info on sentiment analysis,
keywords, tokens, and the number of shares of each article. We treat
this as a classification problem with more than 1400 shares as popular
and less than 1400 shares being unpopular. We use two methods to predict
popularity, boosted trees and logistic regression. We then compare these
models at the end of the article.

# Setup

## Importing

We import the following packages for use thoughout our analysis.

  - tidyverse-Data processing
  - GGally-Creating scatterplot/correlation features
  - caret-Model contruction
  - MASS-Model Selection
  - pROC-Plotting ROC curve for logistic regression.
  - stargazer-Making the regression results pretty.  
  - knitr- Making pretty tables in R.

Given we have a large amount of observations we want to use as many
variables as possible. There are 60 different variables in the dataset
with the following descriptions taken from [this
website](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

0.  url: URL of the article (non-predictive)
1.  timedelta: Days between the article publication and the dataset
    acquisition (non-predictive)
2.  n\_tokens\_title: Number of words in the title
3.  n\_tokens\_content: Number of words in the content
4.  n\_unique\_tokens: Rate of unique words in the content
5.  n\_non\_stop\_words: Rate of non-stop words in the content
6.  n\_non\_stop\_unique\_tokens: Rate of unique non-stop words in the
    content
7.  num\_hrefs: Number of links
8.  num\_self\_hrefs: Number of links to other articles published by
    Mashable
9.  num\_imgs: Number of images
10. num\_videos: Number of videos
11. average\_token\_length: Average length of the words in the content
12. num\_keywords: Number of keywords in the metadata
13. data\_channel\_is\_lifestyle: Is data channel ‘Lifestyle’?
14. data\_channel\_is\_entertainment: Is data channel ‘Entertainment’?
15. data\_channel\_is\_bus: Is data channel ‘Business’?
16. data\_channel\_is\_socmed: Is data channel ‘Social Media’?
17. data\_channel\_is\_tech: Is data channel ‘Tech’?
18. data\_channel\_is\_world: Is data channel ‘World’?
19. kw\_min\_min: Worst keyword (min. shares)
20. kw\_max\_min: Worst keyword (max. shares)
21. kw\_avg\_min: Worst keyword (avg. shares)
22. kw\_min\_max: Best keyword (min. shares)
23. kw\_max\_max: Best keyword (max. shares)
24. kw\_avg\_max: Best keyword (avg. shares)
25. kw\_min\_avg: Avg. keyword (min. shares)
26. kw\_max\_avg: Avg. keyword (max. shares)
27. kw\_avg\_avg: Avg. keyword (avg. shares)
28. self\_reference\_min\_shares: Min. shares of referenced articles in
    Mashable
29. self\_reference\_max\_shares: Max. shares of referenced articles in
    Mashable
30. self\_reference\_avg\_sharess: Avg. shares of referenced articles in
    Mashable
31. weekday\_is\_monday: Was the article published on a Monday?
32. weekday\_is\_tuesday: Was the article published on a Tuesday?
33. weekday\_is\_wednesday: Was the article published on a Wednesday?
34. weekday\_is\_thursday: Was the article published on a Thursday?
35. weekday\_is\_friday: Was the article published on a Friday?
36. weekday\_is\_saturday: Was the article published on a Saturday?
37. weekday\_is\_sunday: Was the article published on a Sunday?
38. is\_weekend: Was the article published on the weekend?
39. LDA\_00: Closeness to LDA topic 0
40. LDA\_01: Closeness to LDA topic 1
41. LDA\_02: Closeness to LDA topic 2
42. LDA\_03: Closeness to LDA topic 3
43. LDA\_04: Closeness to LDA topic 4
44. global\_subjectivity: Text subjectivity
45. global\_sentiment\_polarity: Text sentiment polarity
46. global\_rate\_positive\_words: Rate of positive words in the content
47. global\_rate\_negative\_words: Rate of negative words in the content
48. rate\_positive\_words: Rate of positive words among non-neutral
    tokens
49. rate\_negative\_words: Rate of negative words among non-neutral
    tokens
50. avg\_positive\_polarity: Avg. polarity of positive words
51. min\_positive\_polarity: Min. polarity of positive words
52. max\_positive\_polarity: Max. polarity of positive words
53. avg\_negative\_polarity: Avg. polarity of negative words
54. min\_negative\_polarity: Min. polarity of negative words
55. max\_negative\_polarity: Max. polarity of negative words
56. title\_subjectivity: Title subjectivity
57. title\_sentiment\_polarity: Title polarity
58. abs\_title\_subjectivity: Absolute subjectivity level
59. abs\_title\_sentiment\_polarity: Absolute polarity level
60. shares: Number of shares (target)

## Partitioning into Test and Training.

We filter the data for each day so that we are only viewing output
pertaining to this day of the week. We choose to classify a shares
binary variable that is 0 for shares\<1400 and 1 for shares\>1400. We
then remove the timedelta and url variables because they are not useful
predictors. Afterwards we separate %70 of our data for training and use
the remaining observations for testing our models accuracy. We also set
our seed so that our results are reproducible.

``` r
params<-list()
params$day<-'monday'
news_day <-news %>%
  filter(get(paste0('weekday_is_',params$day))==1) %>%
  mutate(share_binary=as.factor(ifelse(shares<1400,0,1))) %>%
  dplyr::select(-starts_with('weekday_is'),
                -is_weekend,
                -url,
                -timedelta)

set.seed(628)
index_train<-unlist(createDataPartition(news_day$share_binary,p=.7))
training<-news_day[index_train,]
testing<-news_day[-index_train,]
```

# Vizualizations and Summary Statistics

## Shares

Lets look into the shares variable and look at its distribution. The
first histogram appears to be heavily affected by some outliers in the
shares variable. So we remove outliers for the second visualization. The
classification threshold is also depicted on the graph. The shares
variable has a mean around 1000 shares and appears to be skewed right.

``` r
ggplot(data=training,aes(x=shares))+
  geom_histogram(bins = 100)+
  ggtitle('Histogram of Shares of Article')+
  geom_vline(xintercept=1400)
```

![](sunday_files/figure-gfm/Shares-1.png)<!-- -->

``` r
iqr<-IQR(training$shares)
lower<- quantile(training$shares,.25)-1.5*iqr
upper<- quantile(training$shares,.75)+1.5*iqr

training_out_rem<-training %>%
  filter(shares>lower & shares<upper )

ggplot(data=training_out_rem,aes(x=shares)) +
  geom_histogram(bins = 50)+
  ggtitle('Histogram of Shares of Article with Outliers Removed')+
  geom_vline(xintercept=1400)
```

![](sunday_files/figure-gfm/Shares-2.png)<!-- -->

## Title Characteristics

Sharing an article is a two step process. First, the person must click
on the article. Some factors that may impact the likeliness of a click
are title features, pictures, and the channel type. Lets examine some of
the patterns or lack thereof in our data. We make graphs of the title
sentiment polarity and subjectivity for various channels against the
total shares variables. We also add a line to depict the threshold for
popularity. We also find the mean and standard deviation for these
variables and include them in the table below.

``` r
training_out_rem<-training_out_rem %>%
  mutate(channel=ifelse(data_channel_is_lifestyle==1,'lifestyle',
                       ifelse(data_channel_is_entertainment==1,'entertainment',
                       ifelse(data_channel_is_bus==1,'bus',
                       ifelse(data_channel_is_socmed==1,'socmed',
                       ifelse(data_channel_is_tech==1,'tech',
                       ifelse(data_channel_is_world==1,'world','unknown')))))))

ggplot(training_out_rem,aes(x=title_sentiment_polarity,y=shares,color=title_subjectivity))+
         facet_wrap(~channel)+
         geom_jitter()+
         ggtitle('Title Characteristics by Channel')+
         xlab('Title Subjectivity')+
         ylab('Shares')+
         geom_hline(yintercept = 1400)
```

![](sunday_files/figure-gfm/Data%20Visualizations-Title-1.png)<!-- -->

``` r
training_out_rem %>%
  group_by(channel) %>%
  summarise('Mean Number of Shares'=mean(shares),
            'Number of Articles'=n(),
            'Average Title Sentiment Polarity'=mean(title_sentiment_polarity),
            'Standard Deviation of Title Sentiment Polarity'=sd(title_sentiment_polarity),
            'Average Title Subjectivity'=mean(title_subjectivity),
            'Standard Deviation of Title Subjectivity'=sd(title_subjectivity)) %>%
  kable()
```

| channel       | Mean Number of Shares | Number of Articles | Average Title Sentiment Polarity | Standard Deviation of Title Sentiment Polarity | Average Title Subjectivity | Standard Deviation of Title Subjectivity |
| :------------ | --------------------: | -----------------: | -------------------------------: | ---------------------------------------------: | -------------------------: | ---------------------------------------: |
| bus           |              1671.509 |                719 |                        0.0843820 |                                      0.2234646 |                  0.2435838 |                                0.2884099 |
| entertainment |              1364.016 |                859 |                        0.0675649 |                                      0.2873559 |                  0.2983062 |                                0.3122639 |
| lifestyle     |              1847.813 |                193 |                        0.1004995 |                                      0.2616731 |                  0.2826286 |                                0.3242623 |
| socmed        |              2360.740 |                200 |                        0.1051938 |                                      0.2428283 |                  0.2627411 |                                0.3234653 |
| tech          |              1861.390 |                749 |                        0.0758198 |                                      0.2175264 |                  0.2512278 |                                0.3049750 |
| unknown       |              1807.124 |                482 |                        0.0536421 |                                      0.3142569 |                  0.3411367 |                                0.3561938 |
| world         |              1317.651 |                896 |                        0.0306815 |                                      0.2425522 |                  0.2338387 |                                0.3089401 |

## Global Characteristics

Next lets look at sentiment polarity and subjectivity of the whole
article. We create the same graphs and summary statistics as above but
use the measures for entire text.

``` r
ggplot(training_out_rem,aes(x=global_sentiment_polarity,y=shares,color=global_subjectivity))+
         facet_wrap(~channel)+
         geom_jitter()+
         ggtitle('Title Characteristics by Channel')+
         xlab('Title Subjectivity')+
         ylab('Shares')+
  geom_hline(yintercept = 1400)
```

![](sunday_files/figure-gfm/Data%20Visualizations-Global-1.png)<!-- -->

``` r
training_out_rem %>%
  group_by(channel) %>%
  summarise('Mean Number of Shares'=mean(shares),
            'Number of Articles'=n(),
            'Average Global Sentiment Polarity'=mean(global_sentiment_polarity),
            'Standard Deviation of Sentiment Polarity'=sd(global_sentiment_polarity),
            'Average Global Subjectivity'=mean(global_subjectivity),
            'Standard Deviation of Global Subjectivity'=sd(global_subjectivity)) %>%
  kable()
```

| channel       | Mean Number of Shares | Number of Articles | Average Global Sentiment Polarity | Standard Deviation of Sentiment Polarity | Average Global Subjectivity | Standard Deviation of Global Subjectivity |
| :------------ | --------------------: | -----------------: | --------------------------------: | ---------------------------------------: | --------------------------: | ----------------------------------------: |
| bus           |              1671.509 |                719 |                         0.1371226 |                                0.0805075 |                   0.4352031 |                                 0.0820544 |
| entertainment |              1364.016 |                859 |                         0.1103801 |                                0.1008036 |                   0.4519752 |                                 0.1079260 |
| lifestyle     |              1847.813 |                193 |                         0.1552413 |                                0.0930134 |                   0.4750594 |                                 0.0973770 |
| socmed        |              2360.740 |                200 |                         0.1516057 |                                0.0987823 |                   0.4598300 |                                 0.0960729 |
| tech          |              1861.390 |                749 |                         0.1418428 |                                0.0750818 |                   0.4511638 |                                 0.0753622 |
| unknown       |              1807.124 |                482 |                         0.0997525 |                                0.1229466 |                   0.4351192 |                                 0.1982097 |
| world         |              1317.651 |                896 |                         0.0773059 |                                0.0804323 |                   0.4025898 |                                 0.1060595 |

Lastly lets see how the number of shares varies by number of videos and
images. We filter the number of videos and images so the distributions
are more visible.

``` r
ggplot(training_out_rem %>% dplyr::filter(num_videos<5,
                                          num_imgs<10),aes(x=num_imgs,y=shares,color=num_videos))+
         geom_jitter()+
         geom_smooth(method='lm')+
         ggtitle('Videos by Channel')+
         xlab('Videos and Graphics vs Shares')+
         ylab('Shares')
```

![](sunday_files/figure-gfm/Data%20Visualizations-Grpahics-1.png)<!-- -->

# Modeling

## Boosted Tree Model

We train a boosted tree model below. Repeated 10 fold cross validation
was used to select the model parameters below.

``` r
train.control <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3)

 n.trees<-c(200,300)
 interaction.depth<-c(3,4)
 shrinkage<-c(.05)
 n.minobsinnode<-c(10)
 param_grid<-data.frame(crossing(n.trees,interaction.depth,shrinkage,n.minobsinnode))

 boost_fit = train(share_binary ~ .-shares,
                   data=training,
                   method="gbm",
                   trControl=train.control,
                   distribution="bernoulli",
                   tuneGrid=param_grid,
                   verbose=FALSE)
 
 
boost_fit$results %>% kable()
```

|   | shrinkage | interaction.depth | n.minobsinnode | n.trees |  Accuracy |     Kappa | AccuracySD |   KappaSD |
| - | --------: | ----------------: | -------------: | ------: | --------: | --------: | ---------: | --------: |
| 1 |      0.05 |                 3 |             10 |     200 | 0.6495151 | 0.2984914 |  0.0191220 | 0.0382787 |
| 3 |      0.05 |                 4 |             10 |     200 | 0.6487260 | 0.2969199 |  0.0168792 | 0.0337709 |
| 2 |      0.05 |                 3 |             10 |     300 | 0.6508001 | 0.3010961 |  0.0172123 | 0.0345183 |
| 4 |      0.05 |                 4 |             10 |     300 | 0.6495853 | 0.2987087 |  0.0180022 | 0.0360409 |

``` r
boost_pred <- predict(boost_fit,newdata = dplyr::select(testing, -share_binary))
conf_boost<-confusionMatrix(boost_pred,testing$share_binary)
```

We select the boosted model with the highest accuracy hoping that this
will translate to a high accuracy on our testing dataset. Lets make a
plot of the confusion matrix and look at the accuracy, precision, and
recall.

``` r
fourfoldplot(conf_boost$table)
```

![](sunday_files/figure-gfm/Boosted%20Tree%20Model-Fit-1.png)<!-- -->

``` r
data.frame(Accuracy=conf_boost$overall[1],
           Prevision=conf_boost$byClass[5],
           Recall=conf_boost$byClass[6]) %>%
  kable()
```

|          |  Accuracy | Prevision |    Recall |
| -------- | --------: | --------: | --------: |
| Accuracy | 0.6436436 | 0.6417281 | 0.6207951 |

## Regression Model

It looks like the LDA is not linearly independent from each other. Lets
remove LDA\_04 and continue. We remove high leverage points by removing
all points that have a cooks distance greater than 4/(sample size).

``` r
#We fit a logistic regression model
glmFit <- glm(share_binary ~ . -LDA_04-shares, data = training, family = "binomial")
plot(glmFit,which = 4, id.n = 5)
```

![](sunday_files/figure-gfm/Logistic%20Regression%20Fit-1.png)<!-- -->

``` r
cooksd <- cooks.distance(glmFit)
sample_size <- nrow(training)
influential <- as.numeric(names(cooksd)[(cooksd > (4/sample_size))])
training_no_lev<-training  %>%
  filter(!row_number() %in% influential)
```

Now we do backward model selection using the lowest AIC. Mimizing AIC
helps us to come up with a parsimonious model that still fits the data
well.

``` r
glmFit <- glm(share_binary ~ . -LDA_04-shares, 
              data = training_no_lev,
              family = "binomial")

step.model <- glmFit %>% stepAIC(trace = FALSE)
```

We then use the ROC curve to find the Youden point which maximizes
sensitivity+specificity. This allows us to classify the values returned
from our logistic regression into popular and unpopular.

``` r
roc_obj <- roc(training_no_lev$share_binary, glmFit$fitted.values)
thresh<-coords(roc_obj, "best", "threshold",transpose=FALSE)
plot(roc_obj)
```

![](sunday_files/figure-gfm/Logistic%20Regression%20Threshold-1.png)<!-- -->

Lets look at the estimates of our regression. We see many variables that
are statistically significant.

``` r
summary(step.model)
```

    ## 
    ## Call:
    ## glm(formula = share_binary ~ n_non_stop_words + num_hrefs + num_self_hrefs + 
    ##     num_imgs + num_keywords + data_channel_is_entertainment + 
    ##     data_channel_is_socmed + data_channel_is_tech + data_channel_is_world + 
    ##     kw_avg_min + kw_min_max + kw_max_max + kw_avg_max + kw_min_avg + 
    ##     kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_avg_sharess + 
    ##     LDA_00 + LDA_01 + LDA_02 + LDA_03 + global_subjectivity + 
    ##     rate_positive_words + avg_positive_polarity + avg_negative_polarity + 
    ##     title_subjectivity + abs_title_subjectivity, family = "binomial", 
    ##     data = training_no_lev)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.4973  -1.0110   0.3029   1.0286   2.0813  
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                   -2.042e+00  3.548e-01  -5.755 8.69e-09 ***
    ## n_non_stop_words              -1.022e+00  3.353e-01  -3.049 0.002296 ** 
    ## num_hrefs                      2.144e-02  4.111e-03   5.214 1.84e-07 ***
    ## num_self_hrefs                -2.857e-02  1.012e-02  -2.825 0.004735 ** 
    ## num_imgs                       1.056e-02  5.167e-03   2.043 0.041046 *  
    ## num_keywords                   5.342e-02  2.123e-02   2.516 0.011861 *  
    ## data_channel_is_entertainment -2.133e-01  1.291e-01  -1.651 0.098656 .  
    ## data_channel_is_socmed         1.278e+00  1.928e-01   6.630 3.35e-11 ***
    ## data_channel_is_tech           6.758e-01  1.379e-01   4.899 9.64e-07 ***
    ## data_channel_is_world          3.436e-01  1.671e-01   2.056 0.039740 *  
    ## kw_avg_min                     1.950e-04  1.363e-04   1.431 0.152449    
    ## kw_min_max                    -3.212e-06  1.061e-06  -3.028 0.002460 ** 
    ## kw_max_max                    -7.148e-07  2.086e-07  -3.426 0.000612 ***
    ## kw_avg_max                    -7.286e-07  4.556e-07  -1.599 0.109741    
    ## kw_min_avg                    -7.337e-05  4.545e-05  -1.614 0.106440    
    ## kw_max_avg                    -1.056e-04  2.348e-05  -4.498 6.86e-06 ***
    ## kw_avg_avg                     9.038e-04  9.386e-05   9.629  < 2e-16 ***
    ## self_reference_min_shares      2.549e-05  9.047e-06   2.817 0.004844 ** 
    ## self_reference_avg_sharess     1.776e-05  6.690e-06   2.655 0.007926 ** 
    ## LDA_00                         7.910e-01  2.076e-01   3.810 0.000139 ***
    ## LDA_01                        -3.911e-01  2.445e-01  -1.599 0.109801    
    ## LDA_02                        -8.022e-01  2.482e-01  -3.232 0.001230 ** 
    ## LDA_03                        -5.321e-01  2.157e-01  -2.467 0.013607 *  
    ## global_subjectivity            1.190e+00  4.671e-01   2.547 0.010856 *  
    ## rate_positive_words            6.234e-01  2.464e-01   2.530 0.011406 *  
    ## avg_positive_polarity         -1.161e+00  4.482e-01  -2.591 0.009577 ** 
    ## avg_negative_polarity         -5.473e-01  3.222e-01  -1.699 0.089411 .  
    ## title_subjectivity             2.400e-01  1.242e-01   1.932 0.053397 .  
    ## abs_title_subjectivity         6.819e-01  2.036e-01   3.349 0.000811 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6308.7  on 4552  degrees of freedom
    ## Residual deviance: 5499.3  on 4524  degrees of freedom
    ## AIC: 5557.3
    ## 
    ## Number of Fisher Scoring iterations: 6

Lets test the fit of our final model against our testing dataset.

``` r
test_pred <- predict(step.model,
                    newdata = dplyr::select(testing, -share_binary),
                    type = "response")

conf_exp<-confusionMatrix(data = as.factor(as.numeric(test_pred>thresh$threshold)), reference = testing$share_binary)

fourfoldplot(conf_exp$table)
```

![](sunday_files/figure-gfm/Logistic%20Regression%20Results-1.png)<!-- -->

``` r
data.frame(Accuracy=conf_exp$overall[1],
           PreCision=conf_exp$byClass[5],
           Recall=conf_exp$byClass[6]) %>%
  kable()
```

|          |  Accuracy | PreCision |    Recall |
| -------- | --------: | --------: | --------: |
| Accuracy | 0.6321321 | 0.6062176 | 0.7155963 |

# Comparison of Models

Lets see how the boosted tree compares to the logistic regression model.
Both of the models are fit the data about as well as the other. The
models dont appear to be overfitted or underfitted because

``` r
Accuracy<-c(conf_boost$overall[1],conf_exp$overall[1])
Precision<-c(conf_boost$byClass[5],conf_exp$byClass[5])
Recall<-c(conf_boost$byClass[6],conf_exp$byClass[6])

model<-c('Boosted Tree','Logistic Regression')
data.frame(model,Accuracy,Precision,Recall) %>% kable()
```

| model               |  Accuracy | Precision |    Recall |
| :------------------ | --------: | --------: | --------: |
| Boosted Tree        | 0.6436436 | 0.6417281 | 0.6207951 |
| Logistic Regression | 0.6321321 | 0.6062176 | 0.7155963 |
