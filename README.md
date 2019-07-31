# EMGTimeSeriesAnalysis
Trains models to identify phases in each EMG Time Series and make rating on the subject's proficiency.

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

## Part 1: Ensemble Rating
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

## Part 2: Ensemble Segmenting
### How It Works
### Interpreting the Results

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