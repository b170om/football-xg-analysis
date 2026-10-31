# Football Match Outcome Prediction with xG and Elo

A data science project exploring whether recent form, expected goals (xG), league position, and historical Elo ratings can help predict football match outcomes: **home win, draw, or away win**.

The focus is on building features available **before kickoff**, comparing predictive approaches, and evaluating the selected model on a later, unseen season.

## Dataset

- **Source:** [Football Data: Expected Goals and Other Metrics (Kaggle)](https://www.kaggle.com/datasets/slehkyi/extended-football-stats-for-european-leagues-xg)
- **Coverage:** Six European football leagues, 2014/15–2018/19 seasons
- **Match-level records:** 10,330 matches after merging home and away team observations
- **Target:** `w` (home win), `d` (draw), or `l` (away win)

The original dataset contains separate records for each team's performance in a match. These are combined into one match-level record for prediction.

## Approach

### Feature engineering

All predictive features are constructed from information available prior to the match:

- **Recent form:** Rolling averages of points, goals scored, and goals conceded over the previous five matches
- **Expected performance:** Rolling xG, xG against, and expected points (xPTS)
- **League position:** Pre-match standings and differences between the two teams
- **Elo ratings:** Historical opponent-adjusted strength, updated chronologically after each result

### Modeling and validation

Logistic Regression models with standardized features were compared with a home-win baseline and alternative models, including Random Forest. Feature sets were expanded progressively to examine their contribution.

To avoid evaluating on the data used for model selection, the dataset was split chronologically:

| Split | Seasons | Purpose |
| --- | --- | --- |
| Training | 2014/15–2016/17 | Model fitting |
| Validation | 2017/18 | Model selection and Elo tuning |
| Test | 2018/19 | Final held-out evaluation |

The Elo experiment tested **16 combinations** of update factors (`K = 10, 20, 32, 40`) and home-advantage values (`0, 40, 65, 90`). The selected configuration was **K = 32** and **home advantage = 65**. The final model was retrained on the training and validation seasons together before test evaluation.

## Results

### Validation model comparison (2017/18)

| Model | Accuracy | Macro F1 | Log Loss |
| --- | ---: | ---: | ---: |
| Always predict home win (baseline) | 44.92% | — | — |
| Logistic Regression — Recent Form | 50.07% | 0.358 | 1.010 |
| Logistic Regression — Expected Performance | 51.05% | 0.366 | 1.002 |
| Logistic Regression — Combined | 51.73% | 0.376 | 0.998 |
| Logistic Regression — Combined + League Position | 51.82% | 0.380 | 0.991 |
| Random Forest — Combined | 51.00% | 0.363 | 1.001 |
| Logistic Regression — Combined + Initial Elo (K=20, H=65) | 54.06% | 0.401 | 0.978 |
| **Logistic Regression — Combined + Tuned Elo (K=32, H=65)** | **54.25%** | **0.404** | **0.977** |

The comparison shows that adding strength-gap features and historical Elo ratings to the combined feature set improved validation accuracy from **51.82% to 54.25%**. Because strength gaps and Elo were introduced together in this comparison, the improvement cannot be attributed to Elo alone. Tuning the Elo parameters provided only a small additional gain over the initial configuration (54.06%). The leading parameter combinations performed nearly identically.

### Additional experiment: Promoted teams' Elo ratings

We tested assigning lower initial Elo ratings to newly promoted teams, assuming they might be weaker than established top-division teams. However, this approach reduced validation accuracy, so we retained the original Elo initialization strategy.

### Final held-out test (2018/19)

| Metric | Result |
| --- | ---: |
| Accuracy | **52.14%** |
| Macro F1 | 0.386 |
| Log Loss | 0.984 |
| Always-predict-home-win baseline | 44.65% |
| Test matches | 2,056 |

The final model was retrained on the training and validation seasons together, then evaluated on the unseen 2018/19 season. It outperformed the test baseline by **7.49 percentage points** in accuracy. Validation performance was used for model selection; test results were not used to tune the model.

### Performance by league (2018/19 test season)

We also evaluated the final model separately for each league, without retraining or adjusting its parameters. The chart compares test accuracy with the home-win baseline in each league.

![Final model accuracy compared with the home-win baseline across leagues](figures/leag   ue_performance.png)

| League | Matches | Model accuracy | Home-win baseline |
| --- | ---: | ---: | ---: |
| English Premier League | 377 | **58.6%** | 47.7% |
| Bundesliga | 304 | 54.3% | 45.1% |
| Serie A | 379 | 52.5% | 43.8% |
| Russian Premier League | 239 | 52.3% | 43.1% |
| Ligue 1 | 379 | 48.0% | 43.3% |
| La Liga | 378 | 47.6% | 44.4% |

The **English Premier League had the highest test accuracy (58.6%)**, improving on its baseline by **10.9 percentage points**. The model outperformed the home-win baseline in all six leagues, but improvements varied considerably. Since these comparisons reflect a single test season, they do not establish that the model will consistently perform better in one league than another.

### Main limitation: draws

The final model predicted only **one draw** among 2,056 test matches and did not identify any of the **539 actual draws** correctly. The relatively modest Macro F1 score reflects this weakness. Accuracy alone does not fully describe performance across all three match outcomes.

## Reproduce the analysis

1. Download `understat_per_game.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/slehkyi/extended-football-stats-for-european-leagues-xg).
2. Place the CSV at `data/understat_per_game.csv` in the project root.
3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open `notebooks/football_analysis.ipynb` in Jupyter or VS Code and run its cells from top to bottom.

The notebook reads the CSV with a relative path (`../data/understat_per_game.csv`), so run it from the `notebooks` directory or use a notebook environment that sets that directory as the working directory.

## What I Learned

This project gave me hands-on experience with the full data science workflow, from raw data to evaluating a predictive model.

1. **Working with real-world data:** I learned how to clean, organize, and transform raw football data, including combining team-level records into individual matches and preparing datasets for machine learning.

2. **Preventing data leakage:** I learned the importance of using only information available before each match. This required careful handling of historical features and chronological train, validation, and test splits.

3. **Feature engineering and domain knowledge:** I explored how football knowledge can be translated into meaningful predictive features, including recent team form, expected goals (xG), league positions, and Elo ratings.

4. **Model experimentation and evaluation:** I gained experience comparing machine learning models, testing different feature combinations, and tuning hyperparameters. I also learned that more complex models do not necessarily perform better.

5. **Critical thinking and problem-solving:** I learned to question assumptions, identify potential edge cases, and look for practical solutions rather than simply accepting the first result. Working with real data taught me to investigate unexpected behavior and think carefully about how each decision affects the analysis.

6. **Turning theory into practice:** One of the most valuable parts of this project was applying concepts I had studied at university to a real-world problem. It helped me connect theoretical knowledge in statistics, machine learning, and data engineering with the challenges of building an actual predictive system.

7. **Critically interpreting results:** I learned to look beyond accuracy and evaluate models using Macro F1, Log Loss, and confusion matrices. Although the final model outperformed the baseline, its difficulty predicting draws highlighted the importance of understanding and communicating model limitations.

## Potential next steps

Investigate goal-based approaches (for example, Poisson models) and better treatment of draw probabilities. Any further model changes would require a new untouched evaluation period to provide an independent performance estimate.
