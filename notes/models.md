## Preprocessors

### Decomposition

- **PCA** nggak dibutuhkan kalau udah seleksi fitur via RFE


## Regressors

### Linear models

- **Lasso** jangan pernah ngeset `alpha = 0` ntar hangs
- **LogisticRegression** berat banget bikin hang untuk data segede ini
- **TheilSenRegressor** berat banget bikin hang

### Trees/Forests

- **ExtraTreesRegressor** kadang **jauh** lebih bagus dari **RandomForestRegressor**
