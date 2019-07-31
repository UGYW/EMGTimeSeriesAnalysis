# EMGTimeSeriesAnalysis
The project is structured as follows:
* **Rating Prediction**:
Given times series and their ratings,
construct a model that will predict the rating of a given time series.
* **Timestamp Prediction**:
Given time series and timestamps for events that occurred in each of these time series,
construct a model that will predict the timestamps of these events of a given time series.

For each of these models, there are two data categories available - ROB and LAP.
They are structured exactly the same, but their sources were different.
ROB data refers to the robotic data, and LAP data refers to the laparoscopic data.

**Who is this README meant for?**

This is a high-level overview of the analysis model.
For further troubleshooting, refer to the Troubleshooting Guide in the Drive.

## Requirements
1. **Python 3.7** or above
    + Most current online distributions (Kaggle, Google Colab., etc) use Python 3.6 or below,
    so sometimes a few unexpected errors can occur.
    They're not catastrophic though, and can easily be fixed.
    It's more of a nuisance than anything.
2. **tslearn**
    + Can be installed with pip via `pip install tslearn` or
    `conda install -c conda-forge tslearn` if you have Anaconda or Miniconda.
    + You can find their repository [here](https://github.com/rtavenar/tslearn)
    and the documentation [here](https://tslearn.readthedocs.io/en/latest/)

All the other packages invoked in this project are generally
already found with the Python distribution by default,
such as numpy and random.

## Running the Program
Here is a list of files that can be executed via Python.
(I.E run through `python <nameofprogram>`)

* **EMGDataManagerPrototype**:
This is just for debugging possible errors within the Data Manager.
It does nothing besides preprocess the data in the `data_sample` directory.
You can modify the arguments within that file to refer to other directories as well.
* **EnsembleRatingPrototype and EnsembleSegmentingPrototype**:
Similarly, this is for simple debugging of the two analysis systems.
Unlike `EMGDataManagerPrototype`, they do not need arguments to sample data,
and come with small toy datasets.
If you would like to learn the flow of those programs, do start there.
* **EnsembleRating and EnsembleSegmenting**:
These are the actual programs that execute the respectively titled tasks.
Note that, similarly to `EMGDataManagerPrototype`,
you need to initialize the directory to the available data.
You can see more on how to inspect their output below.

## Ensemble Rating
### How It Works
The model pipeline looks like this.
```
ROB DATA -> ROB MUS1 DATA -> SVR MODEL -> MUS1 RATING ->       ENSEMBLE
         -> ROB MUS2 DATA -> SVR MODEL -> MUS2 RATING ->     MODEL & RATING
         -> ...                                                  (ROB)
         -> ROB MUS6 DATA -> SVR MODEL -> MUS6 RATING ->
ROB TIMES ----------only used in ensemble-------------->

LAP DATA -> LAP MUS1 DATA -> SVR MODEL -> MUS1 RATING ->       ENSEMBLE
         -> LAP MUS2 DATA -> SVR MODEL -> MUS2 RATING ->     MODEL & RATING
         -> ...                                                  (LAP)
         -> LAP MUS6 DATA -> SVR MODEL -> MUS6 RATING ->
LAP TIMES ----------only used in ensemble-------------->
```
The data is split into three portions - *training, interim*, and *test*.

The *training* data is used for training the individual models,
the *interim* data is used for training the ensemble model,
and the *test* data is used for assessing the performance of the model.

**Why use interim data?**

The training data is used to train the specific models.
If we were to use the said data
and make the model predict on them to produce the labels for the ensemble training,
that would not be proper practice,
since the individual models would have already
accessed the actual ratings in order to train themselves in the first place.


In other words, the overall model would have been exposed to
the actual ratings before the final step, which would be cheating.

**Why SVR?**

The ratings are continuous values rather than discrete categories,
so we needed a regressor from the tslearn package.
As far as we could tell, the only regressor there was the State Vector Regressor (SVR),
the regression version of the State Vector Machine (SVM).

*(Basically, it was the only option if we wanted to use the tslearn package)*

Given the timespan we had for this project,
I decided we could do the most if we took advantage of tslearn
instead of trying to build my own process with scikit.

### Interpreting the Results
These are raw values from predicting
the rating of the LAP datasets within FRMU and GEES.
```
LAP Prediction
[53.69444444 60.93981481 61.42592593] <--- These are the predicted ratings for three knots
LAP Actual
[53.0, 52.5, 56.5] <--- These are the actual ratings for the same three knots
```
You can calculate additional indicators such as:
```
LAP (Absolute) Difference
[0.69444444 8.43981481 4.92592593] <--- Produced using the score function
LAP Diff Total
14.060185185185176 <--- summed from the result of the score
```
You can also modify the score function to calculate additional indicators.

## Ensemble Segmenting
### How It Works
Given some time series `t = [t1, t2 ... ]`, we look for a group of other time series
who are the closest to it.

*The assumption here is that time series that are similar
will also have similar timestamps.*

We then predict the timestamps of time series `t` to be an average of said group.

#### Known Issue: The Wrong Metric?
In other words, **using DTW as the metric may be incompatible with said assumption**,
since we don't know how to un-warp these timestamps.

For example, if Person A and Person B had a similar waveform shape
(i.e high similarity via DTW),
but Person B did it at twice the speed,
it would be erroneous to say that Person A would have similar timestamps than Person B.

Unfortunately, it's also a known issue within tslearn that KNN
does not work with any other metric.

There is a possible solution in that tslearn is capable of calculating the raw DTW metric between two time series,
and that may be used to un-warp (e.g indicate that Person B did the knot at twice the speed).
This is a subtle point and may require more testing.

On the other hand, it wouldn't be completely incorrect to say that
people who had similar progressions (i.e high DTW similarity)
probably had similar timestamps as well.

### Interpreting the Results
This is what it looks like raw (without formatting).
```
ROB Prediction
[array([   0.        ,    7.33333333,   14.33333333,   23.16666667,
         28.08333333,   30.58333333, -252.75      , -247.5       ,
         61.08333333,   75.75      ,   87.16666667,   96.66666667,
        108.33333333,  121.16666667,  128.33333333,  136.91666667,
        147.16666667]), array([   0.        ,    7.25      ,   14.08333333,   22.91666667,
         27.91666667,   31.16666667, -253.        , -247.75      ,
         61.08333333,   75.83333333,   88.16666667,   97.58333333,
        109.08333333,  121.5       ,  128.33333333,  137.        ,
        146.83333333])]
ROB Actual
[array([  0,   4,   9,  13,  18,  21,  30,  34,  39,  65,  75,  80,  85,
        96, 103, 112, 116]), array([  0,   4,  11,  16,  21,  26,  35,  43,  47,  59,  73,  80,  98,
       112, 122, 130, 143])]
```
Let's take a look at the first pair in more detail:
```
These are the predicted timestamps of a single time series,
    where each item represents the timestamp of an action
[   0.        ,    7.33333333,   14.33333333,   23.16666667,
         28.08333333,   30.58333333, -252.75      , -247.5       ,
         61.08333333,   75.75      ,   87.16666667,   96.66666667,
        108.33333333,  121.16666667,  128.33333333,  136.91666667,
        147.16666667]
These are the actual timestamps, in milliseconds as well.
[  0,   4,   9,  13,  18,  21,  30,  34,  39,  65,  75,  80,  85, 96, 103, 112, 116]
```

#### Known Issue: Negative Timestamps
See troubleshooting guide for more details.

## Data Downsampling
### Why downsample?
The original data features time series that are over a million data points long.
This slows down the analysis significantly, especially during ensemble segmenting.

### How is the downsampling done?
Each time series is interpolated to be the size of the shortest time series
divided by a factor of 100 or 1000.

## Troubleshooting
1. Why is it so slow?
    + **Rating**:
    Rating is relatively faster at making predictions during the test phase,
     but slower during the training compared to Segmenting.
     Thus, the bottleneck is during the training phase.
     That can be tuned using the parameters in the SVR model,
     but in general not much can be done without switching to a different model.
    + **Segmenting**:
    Segmentation is the opposite of the Rating task, as it's fast during the training phase
    (it only takes enough time to load the data into the model,
    but no significant operation is applied) but slow during test.
    There is a significant bottleneck during the test phase, as each test point
    needs to be compared with a large subset of training data iteratively,
    and if the training data is large, this process becomes very very slow.
    + FYI, Google Colab will shut off after 12hrs, even if there is a process running.