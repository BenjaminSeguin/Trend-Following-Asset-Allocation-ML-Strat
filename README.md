Hi, this project was realized with 3 teammates for a class.
All the output plots generated from the run_strategy file are in the Outputs folder. Aside from that, just run run_strategy and you are all set!

The following is a quick description of the project. Please reach out if you would like the full report !
Enjoy your reading :)

This work is a combination of a top-down approach across asset classes, and aims to capture trends within each asset class.
This fully rule-based strategy is designed for investors with a medium-to-high risk appetite, seeking global exposure across both geographies and asset classes, and who believe in the weak form of the efficient market hypothesis.

To begin, we compute eight trading signals based on technical analysis indicators. More specifically, we work with
the 22-days return, the 65-days return, the exponential moving average (EMA), the 65-days moving average convergence/
divergence (MACD), the 260-days MACD, the 22-day Bollinger bands and the 65-days breakout signal.

They serve as the features in our machine learning models, and we respectively define
them as X = {X1,X2, ...,X8}, and Y as the asset returns per month.

We then train two distinct machine learning models using these signals: Ridge regression and Random Forest.
On the one hand, the Ridge regression provides us with an expected return for each asset per month,

The hyperparameter α controlling the strength of the regularization
is found by performing a grid search using 5-fold cross-validation, and by selecting the value with the best crossvalidated
R2 score in the following objective function.

On the other hand, the Random Forest model is used in a classification framework, and generates the probability of
an upward movement for each asset per month.
The hyperparameters (i.e. the number of estimators, the depth of the tree, the number of samples required to split
an internal node, the number of sample required to be at a leaf node, and the class weight) are found by performing
a grid search using 5-fold cross-validation and by selecting the value with the best accuracy.

Next, we convert our predictions into z-scores, cross-sectionally within each asset class. In other words, for each
asset, we subtract the mean predicted value of all assets in the class during that month from the asset’s prediction
and divide by the standard deviation of those values for the same month. This process is repeated for each asset,
across all months, and for both the Ridge regression and Random Forest models. We then sum the z-scores from
both models to produce one final z-score per asset per month.

After calculating the z-scores, we rank the assets within each asset class based on their respective z-scores,
with higher values indicating better prospects. We determine the asset weights to be the inverse of these rankings,
which is central to our strategy, ensuring that more capital is allocated to assets with the best predicted performance.

The final layer of our strategy involves a macroeconomic overlay. Based on key macroeconomic criteria, we
compute the weights allocated to each asset class for every month. To do so, we work with the Real GDP (YoY),
the CPI (YoY), the unemployment rate and the yield curve as our features. We proceed firstly by identify different
macroeconomic environments by assuming the existence of 4 macroeconomic regimes and by applying K-means
clustering on quarterly data starting in 1954. Hereon, we define an adjustment rule in asset class weights for each
cluster or regime. For instance, cluster 0 corresponds to a regime with strong growth (high GDP, moderate inflation
and unemployment, and positive yield curve), for which the asset class allocation will overweight equities. The
final weights for each asset are obtained by multiplying the asset-specific weights by the corresponding asset class
weights.
To summarize our strategy and backtest methodology, at the end of each month, we calculate signals for each
asset across all asset classes. We then use these signals in our Ridge and Random Forest models to predict the next
month returns and the probability of up movement respectively. Hereon, we rank the assets and construct a portfolio
within each asset class. Finally, we use the latest macroeconomic data to estimate the economic regime we are in and 
apply the asset class deviations accordingly to obtain our portfolio to hold for the next month.

