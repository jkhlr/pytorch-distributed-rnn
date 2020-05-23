from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class CoronaDataProcessor:
    feature_columns = [
        "Population",
        "DaysSince0",
        "ConfirmedCases",
        "Fatalities",
        "DaysSinceFirstDeath",
        "DaysSinceFirstCase",
        "CumulativeConfirmedCases",
        "CumulativeFatalities"
    ]
    label_column = "ConfirmedCases"

    def __init__(self, window_size):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def process_data(self, csv_path):
        raw_data = pd.read_csv(csv_path, parse_dates=['Date'])
        return self.__transform_data(raw_data)

    def __transform_data(self, raw_data, is_test_data=False):
        # Drop columns we are not interested in
        # TODO: We could calculate the weight again
        raw_data = raw_data.drop(["County", "Province_State", "Weight"], axis=1)

        # Sum target and population for a country_region -> we get 2*114 entries for each country/region
        raw_data = raw_data.groupby(["Date", "Country_Region", "Target"]).sum().reset_index()

        # add current cases and deaths into each category
        raw_data = self.__create_target_value_columns(raw_data)

        # discard duplicates and remove columns that are no longer needed
        raw_data = raw_data.groupby(["Country_Region", "Date"]).first().reset_index()
        raw_data = raw_data.drop(["TargetValue", "Target"], axis=1)

        # create cum sums
        raw_data = self.__create_cum_targets(raw_data)

        # create additional day information
        raw_data = self.__add_day_features(raw_data)

        end_indices, start_indices = self.__create_indices(raw_data)

        X = raw_data[self.feature_columns].to_numpy()
        Y = raw_data[self.label_column].to_numpy()
        X = self.__scale_features(X, is_test_data)

        X_train, y_train = self.__create_windows(X, Y, start_indices, end_indices)
        y_train = y_train.view(-1, 1)
        return X_train, y_train

    def __create_cum_targets(self, raw_data):
        raw_data[["CumulativeConfirmedCases", "CumulativeFatalities"]] = raw_data.groupby("Country_Region")[
            ["ConfirmedCases", "Fatalities"]].cumsum()
        return raw_data

    def __add_day_features(self, raw_data):
        dates = raw_data.set_index("Country_Region")["Date"]
        raw_data["DaysSinceFirstCase"] = ((dates - raw_data.loc[raw_data["ConfirmedCases"] > 0]
                                           .groupby("Country_Region")["Date"].min())
                                          .reset_index(drop=True).dt.days)
        raw_data["DaysSinceFirstDeath"] = ((dates - raw_data.loc[raw_data["Fatalities"] > 0]
                                            .groupby("Country_Region")["Date"].min())
                                           .reset_index(drop=True).dt.days)
        raw_data["DaysSince0"] = (raw_data["Date"] - raw_data["Date"].min()).dt.days
        raw_data[["DaysSinceFirstCase", "DaysSinceFirstDeath"]] = raw_data[
            ["DaysSinceFirstCase", "DaysSinceFirstDeath"]].fillna(-1)
        return raw_data

    def __create_indices(self, raw_data):
        start_indices = raw_data.reset_index(drop=False).groupby("Country_Region").first()["index"].to_numpy()
        end_indices = raw_data.reset_index(drop=False).groupby("Country_Region").last()["index"].to_numpy()
        return end_indices, start_indices

    def __scale_features(self, X, is_test_data):
        if is_test_data:
            return self.scaler.transform(X)
        return self.scaler.fit_transform(X)

    def __create_target_value_columns(self, raw_data):
        column_names = ["ConfirmedCases", "Fatalities"]
        temp = raw_data.reset_index().set_index(["Date", "Country_Region"])
        temp[column_names] = (raw_data.groupby(["Date", "Country_Region", "Target"])["TargetValue"]
                              .sum().unstack(level=2))
        raw_data[column_names] = temp.set_index("index")[column_names]
        return raw_data

    def __create_windows(self, features, labels, start_indices, end_indices):
        X = []
        Y = []
        for start_index, end_index in zip(start_indices, end_indices):
            for i in range(start_index, end_index, self.window_size):
                if i + self.window_size > end_index:
                    continue
                train_seq = features[i:i + self.window_size]
                train_label = labels[i + self.window_size]
                X.append(torch.FloatTensor(train_seq))
                Y.append(train_label)
        return torch.stack(X), torch.FloatTensor(Y)
