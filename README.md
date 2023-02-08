# WeMoms

## Installation

Setup your virtual environment using `pyenv` ([pyenv installer](https://github.com/pyenv/pyenv-installer))

```bash
pyenv install 3.10.7
pyenv local 3.10.7
python -m venv venv
source venv/bin/activate
```

Then install the requirements and the package locally


```
make requirements
```

The file `WeMoms_MLE_hiring_test_2023.json.gzip` must be in the folder `data/raw`

## Problem description

### Context

Rank articles published at day `d-1` for each user and return the top 10.

### Metric

- Business Metric / Online Metric

Click Through Rate: `#opened articles / #displayed articles`

- ML Metric / Offline Metric

MAP@10

### Data

#### Unique Id

`trackable_id` group the articles saw by a user `user_id` at time
`tracker_created_at`

####  Label

`has_been_opened` is `True` if the article has been opened

#### Raw Features

- User related features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **user_is_mom** | BOOLEAN | is the user already a mom (she has at least one child) |
| **user_is_pregnant** | BOOLEAN | is the user pregnant |
| **user_is_trying** | BOOLEAN | is the user trying to conceive a baby |
| **user_country_code** | STRING | |
| **days_since_user_account_creation** | INTEGER | Number of days since the user created her WeMoms account |
| **user_pregnancy_current_day** | INTEGER | For pregnant users : number of days in her pregnancy |
| **user_pregnancy_current_week** | INTEGER | For pregnant users : number of weeks in her pregnancy |
| **user_pregnancy_current_month** | INTEGER | For pregnant users : number of months in her pregnancy |
| **user_pregnancy_current_trimester** | INTEGER | For pregnant users : number of trimesters in her pregnancy |
| **user_age** | INTEGER | age of the user (years) |
| **user_is_trying_since_days** | INTEGER | Number of days since the user has started to try to conceive |
| **user_department** | STRING | |
| **user_followings_count** | INTEGER | Number of followers of the user |
| **user_posts_count** | INTEGER | Number of past posts by the user |
| **user_received_comments_count** | INTEGER | Number of comments on users post |
| **user_pictures_count** | INTEGER | Number of pictures posted by the user |
| **user_likes_count** | INTEGER | Number of likes given by the user |
| **user_children_count** | INTEGER | Number of children of the user |
| **user_children_age_month** | FLOAT | Ages of user's children in months |

- Article related features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **post_age_in_minutes** | INTEGER | Ellasped time between post creation and time of the tracked event |
| **author_children_count** | INTEGER | Number of children of the post author |
| **author_maternity_stage** | STRING | Maternity Stage (MS) of the post author (Mom / Pregnant / TTC = Trying to conceive) |
| **author_amenorrhea_week** | INTEGER | Number of amenorrhea weeks of pregnancy for pregnant post authors |
| **post_comments_count** | INTEGER | Number of comments of the post at the time of the event |
| **post_likes_count** | INTEGER | Number of likes of the post at the time of the event |
| **survey_answers_count** | INTEGER | Number of answers of the post at the time of the event if it's a survey |
| **has_picture** | BOOLEAN | Does the post contains a picture ? |
| **has_text** | BOOLEAN | Does the post contains text ? |
| **has_video** | BOOLEAN | Does the post contains a video ? |
| **direct_reports_count** | INTEGER | Number of time a post has been reported at the time of the event |
| **author_children_age_month** | FLOAT | Ages of author's children in months |
| **author_children_age_year** | FLOAT | Ages of author's children in years |
| **author_department** | STRING | |
| **author_age** | FLOAT | |

- Contextual features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **platform** | STRING | ios/android |

- Label

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **has_been_opened** | BOOLEAN | Has the post been opened by the user ? |


## Commands

### Help

```bash
python -m wemoms_homework
```

Will display all possible commands with their description. You can display each command documentation with:

```bash
python -m wemoms_homework <command> --help
```

### Dataset Creation

Using the raw data we want to make a train/validation/test split based on the column `tracker_created_at`
For the default values you can do:

```bash
make dataset
```

The train and validation set are all the lines in the time rage specified in the configuration file.

For the testset, as we want to evaluate the ranking of yesterday's posts, we only keep the opened post from yesterday and add the synthetic "negative" examples.

### Feature Engineering

We want to be able to compute our features separatly by "category". It enables to compute them in parallel and join them only at the end.

To build every group of features run:

```
make build-features
```

and to merge them into a unified parquet file launch:

```
make merge-features
``` 

### Train the model

The Logistic Regression is implemented using Tensorflow to be able to visualize easily the training process using Tensorboard, to save and use the model quickly and to be able to complexify it without changing too much the code. It ease also the normalization of numerical features and the handling of categorical features as it will be embed in the graph. 

```bash
make train
```

### Make predictions

Save the predictions and print the performance

```bash
make predictions
```

### Make tests

```bash
make tests
```

or

```bash
pytest tests
```

## Future Work

 - Dockerfile
 - Complete unit tests
 - More docstring
 - Implement a `Trainer` class to be able to change easily the library which makes the training and the prediction
 - remove duplicate code for feature generation


## Alternatives

Instead of tweaking the model by changing the neural network by a XGBoost, or the pointwise loss by a pairwise loss, we could totally change the way of seeing the problem.

We could view this problem as a sequence of posts read where we want to predict the next most probable item. This type of problem is called `session-based recommendation` and we could use the package [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) of NVIDIA to solve it.

One of its advantages is how fast it can capture user's preferences switches as it observes the past sequence of items. So for pregnant women, where the evolution of the baby and thus the posts read is constantly changing, it could be better to use this approach.

However these kind of algorithm needs lots of data, can be complex to implement and have difficulties with short sessions (cold start), but there are ways to handle this last issue (heuristics at the beginning for instance).

## Deployement

Let's say we already have a data warehouse where the event are stored and updated live if needed.
The different steps for a complete (Kubeflow) pipeline would be:

- Feature Engineering: from the data warehouse, construct all the feature needed. It could be done with SQL queries (for instance using BigQuery's power) or Beam with Spark to have a more sustainable codebase. The features would be also store in databases like SQL for the training part, or in file like parquet or TFRecords. For the serving part, we could use other table depending on the use case, like BigTable.

- Model Training: once we have our features, if we use a neural network, we could deploy our code in a kuberntes pod using our Dockerfile, and launch the training. The evolution of training could be follow using the Tensorboard. At the end the model would be saved in GCS or S3 to be call later.

- Offline evaluation: once our model is trained, we want to observe its performances on a testset. We could use the `ML.PREDICT` function of BigQuery to apply our model to our testset stored in BigQuery and then plug a Google Data Studio on the results to have a proper dashboard. With this, we could easily compared multiple algorithms on the same dashboard and choose the best one to put in production.

- Serving: the serving could be done in two different ways. Either you make your predictions in batch every night and store them in a big key-value database which will be called each time a user comes. Or the predictions are computed online.
  - Offline approach: with few post and not too many users, this could be a good solution with very good response time. The generation of the feature for the testset will be very easy as it would be exactly the same code as the traiset generation. The only problem would be the lack of contextual feature: time of the day, device, live popularity.
  - Online approach: slower (compute the feature then query the model) and more complicated to maintain (multiple services), this approach will be usually more accurate and would use every live and contextual signals. One problem will be the computation of the features live which can be tricky.

- Online evaluation
  - Monitor the services: using grafana dashboard we could observe the number of users coming on the homepage, the number of query to our services (feature construction, predictions) and their response time, the total response time, the ressources usage.
  - Monitor the performances: we can monitor live the business metric (CTR), but also other metrics like the mean rank or the MAP@k the clicked post to relate to the offline metrics.
  - AB Test: to assess the performance of our approach we have to perform an AB Test which will compare our metric to the previous version of the homepage. It has to be well calibrated to be able to make valuable conclusion (randomization unit, sample size, etc.). The conclusions should be made at the right moment: we should be careful with the first days results and wait to have statistical significance.
