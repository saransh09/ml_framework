import pandas as pd
from sklearn import model_selection

"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout 
"""


class CrossValidation:
    def __init__(
            self, 
            df, 
            target_cols, 
            problem_type="binary_classification",
            multilabel_delimeter = ",",
            num_folds=5,
            shuffle,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targest = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimeter = multilabel_delimeter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1

    def split(self):

        if self.problem_type in ["binary_classification", "multiclass_classification"]:
            if self.num_targest != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[self.target_cols[0]].nunique()
            if unique_values==1:
                raise Exception("Only one unique value found in the target class")
            elif unique_values>1:
                # In this case we are using Stratified K-Fold classification
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)

                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target])):
                    self.dataframe.loc[val_idx, "kfold"] = fold
        
        #TODO: The distribution of the folds should be similar, so try and solve this
        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targest != 1 and self.problem_type=="single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targest < 2 and self.problem_type=="mulit_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        # Holdout is important in the TimeSeries datasets
        # When you have a lot of samples, then you cannot afford to do 5 fold / n fold validation
        # In that cases as well we prefer to use holdout
        elif self.problem_type.startswith("holdout_"):
            # holdout_5, holdout_10, ----
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * float(holdout_percentage) / 100)
            self.dataframe.loc[: (len(self.dataframe) - num_holdout_samples), "kfold"] = 0
            self.dataframe.loc[(len(self.dataframe) - num_holdout_samples):, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targest != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x : str(x).split(self.multilabel_delimeter))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets):
                self.dataframe.loc[val_idx, "kfold"] = fold

        else:
            raise Exception("Problem type not understood")


        return self.dataframe


if __name__ == '__main__':
    # Note do not shuffle time series data
    df = pd.read_csv("../input/train.csv")
    cv = CrossValidation(df, target_cols=["target"], problem_type="holdout_20")
    df_split = cv.split()
    print(df_split.kfold.value_counts())