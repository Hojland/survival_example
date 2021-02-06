# WTTE-TVC-XGBOOST

Weibull Time To Event: Time-varying covariates xgboost

A less hacky machine-learning framework for churn- and time to event prediction.
Forecasting problems as diverse as server monitoring to earthquake- and
churn-prediction can be posed as the problem of predicting the time to an event.
WTTE-RNN is an algorithm and a philosophy about how this should be done.

* [blog post](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/)
* [master thesis](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf)
* Quick visual intro to the [model](https://imgur.com/a/HX4KQ)

# Ideas and Basics

You have data consisting of many time-series of events and want to use historic data
to predict the time to the next event (TTE). If you haven't observed the last event
yet we've only observed a minimum bound of the TTE to train on. This results in
what's called *censored data* (in red):

![Censored data](./readme_figs/data.gif)

Instead of predicting the TTE itself the trick is to let your machine learning model
output the *parameters of a distribution*. This could be anything but we like the
*Weibull distribution* because it's
[awesome](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/#embrace-the-Weibull-euphoria).
The machine learning algorithm could be anything gradient-based but we like RNNs
because they are [awesome](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
too.

![example WTTE-RNN architecture](./readme_figs/fig_rnn_weibull.png)

The next step is to train the algo of choice with a special log-loss that can work
with censored data. The intuition behind it is that we want to assign high
probability at the *next* event or low probability where there *wasn't* any events
(for censored data):

![WTTE-RNN prediction over a timeline](./readme_figs/solution_beta_2.gif)

What we get is a pretty neat prediction about the *distribution of the TTE* in each
step (here for a single event):

![WTTE-RNN prediction](./readme_figs/it_61786_pmf_151.png)

A neat sideresult is that the predicted params is a 2-d embedding that can be used to
visualize and group predictions about *how soon* (alpha) and *how sure* (beta). Here
by stacking timelines of predicted alpha (left) and beta (right):

![WTTE-RNN alphabeta.png](./readme_figs/alphabeta.png)


## Warnings

There's alot of mathematical theory basically justifying us to use this nice loss
function in certain situations:

![loss-equation](./readme_figs/equation.png)

So for censored data it only rewards *pushing the distribution up*, beyond the point
of censoring. To get this to work you need the censoring mechanism to be independent
from your feature data. If your features contains information about the point of
censoring your algorithm will learn to cheat by predicting far away based on
probability of censoring instead of tte. A type of overfitting/artifact learning.
Global features can have this effect if not properly treated.



# TODO
[] terraform
[] training data transform
[] implement surv model
[] make postprocessing
[] make some sort of app for it
[] add mlflow to model objects
[] fixing train and score scripts
[] add git name to survival
[] delete survmodel when shap stuff is taken care of
[] look into thomas data
[] also create data for this surv model stuff
[] https://www.ncbi.nlm.nih.gov/books/NBK232486/
[] https://my-ms.org/mri_planes.htm

    def SHAP_force_plot(self, high_pred_n: int = 1,
                        approximate: bool = False):

        X, y = self.split_X_y()
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        X_train = X_train.reset_index(drop=True)
        cat_features = preprocessing_utils.determine_cat(X_train)
        X_train.loc[:, ~X_train.columns.isin(cat_features)] = \
            X_train.loc[:, ~X_train.columns.isin(cat_features)].apply(lambda x: x.round(2))

        X_train[f'{self.y_variable}_pred'] = self.get_predictions(X_train)
        X_train = X_train.sort_values(f'{self.y_variable}_pred',
                          ascending=False).iloc[high_pred_n-1, :]
        idx = X_train.name
        X_train = X_train.iloc[:-1]
        shap_values = self.Predictor.shap_values[idx, :]
        shap.force_plot(np.mean(y_train),
                        shap_values, X_train,
                        matplotlib=True,
                        text_rotation=45)

    def SHAP_force_plot_sim(self, approximate: bool = False):

        X, y = self.split_X_y()
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        idx = np.random.choice(len(X_train), round(len(X_train)/5))
        idx = set(idx)
        idx = np.array(idx)
        shap_values = self.Predictor.shap_values[idx, :]
        X_train = X_train.iloc[idx, :]

        shap.initjs()
        return shap.force_plot(np.mean(y_train),
                               shap_values, X_train)
