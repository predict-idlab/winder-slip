import pandas as pd
import numpy as np
from tsflex.features import (
    FeatureCollection,
    FeatureDescriptor,
    MultipleFeatureDescriptors,
    FuncWrapper,
)
from tsflex.features.utils import make_robust
from tqdm.auto import tqdm
import multiprocess
import scipy.stats as ss
from functional import seq
import re

from pathlib import Path
from typing import List

from sklearn.linear_model import SGDRegressor, HuberRegressor, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

def process_signals(df: pd.DataFrame):
    # based on the formula
    df['Motors.Roller1.WebSpeedSimple'] = df['Motors.Roller1.Speed']*df['AnalogSensors.Radius1Fine']*2*np.pi
    df['Motors.Roller1.SlipSimple'] = df['Motors.Roller1.WebSpeedSimple'] - df['Motors.Traction1.Speed']

    df['Roller1_diff(speed)'] = df['Motors.Roller1.Speed'].diff().fillna(0)
    df['delta_traction_speed'] = df['Motors.Traction1.Speed'] - df['Motors.Traction2.Speed']

    # data from the sensors
    # Traction 1
    df["Motors.Traction1.Setpoint_diff"] = df["Motors.Traction1.Setpoint"].diff().fillna(0)
    df["Motors.Traction1.Position_diff"] = df["Motors.Traction1.Position"].diff().fillna(0)

    df["Traction1_diff(delta(torque, speed))"] = (df["Motors.Traction1.Torque"] - df["Motors.Traction1.Speed"]).diff().fillna(0)
    df["Traction1_delta(torque,diff(speed))"] = df["Motors.Traction1.Torque"] - df["Motors.Traction1.Speed"].diff().fillna(0)
    df["Traction1_diff(speed)"] = df["Motors.Traction1.Speed"].diff().fillna(0)

    # data from the sensors
    # Traction 2
    df["Motors.Traction2.Setpoint_diff"] = df["Motors.Traction2.Setpoint"].diff().fillna(0)
    df["Motors.Traction2.Position_diff"] = df["Motors.Traction2.Position"].diff().fillna(0)
    df["Traction2_diff(delta(torque, speed))"] = (df["Motors.Traction2.Torque"] - df["Motors.Traction2.Speed"]).diff().fillna(0)
    df["Traction2_delta(torque,diff(speed))"] = df["Motors.Traction2.Torque"] - df["Motors.Traction2.Speed"].diff().fillna(0)

    # Dancer
    df["Dancer1_delta(torque,diff(speed))"] = df["Motors.Dancer1.Torque"] - df["Motors.Dancer1.Speed"].diff().fillna(0)
    df["Dancer1_delta(torque, speed)"] = df["Motors.Dancer1.Torque"] - df["Motors.Dancer1.Speed"]
    df["Dancer1_diff(delta(torque, speed))"] = (df["Motors.Dancer1.Torque"] - df["Motors.Dancer1.Speed"]).diff().fillna(0)
    df["Dancer_diff(speed)"] = df["Motors.Dancer1.Speed"].diff().fillna(0)
    df["Dancer_diff(position)"] = df["Motors.Dancer1.Position"].diff().fillna(0)

    # Loadcell
    df["LoadCell1_diff"] = df["AnalogSensors.LoadCell1"].diff().fillna(0)
    df["LoadCell2_diff"] = df["AnalogSensors.LoadCell2"].diff().fillna(0)

    df["delta(LoadCell1, LoadCell2)"] = df["AnalogSensors.LoadCell1"] - df["AnalogSensors.LoadCell2"]
    df["diff(delta(LoadCell1, LoadCell2))"] = df["delta(LoadCell1, LoadCell2)"].diff().fillna(0)

    df["avg_traction1_accel"] = df['Motors.Traction1.Speed'].diff().fillna(0).abs().rolling(10, center=True).mean()
    if TARGET_COL in df.columns:
        df['slip_compensated_acc'] = df['slip'] / (1 + df['avg_traction1_accel'])


    df["avg_traction1_accel"] = (
        df["Motors.Traction1.Speed"]
        .diff()
        .fillna(0)
        .abs()
        .rolling(10, center=True)
        .mean()
    )

    # Gain factor 
    # df["gain_factor"] = -((11 - df["PulseRampTime"]) * df["PulseSpeed"])
    # # Add multiple the gain factor to all current signals
    # for col in df.columns:
    #     if col not in ["gain_factor"] + [TIME_COL, GROUP_COL, TARGET_COL] + ['Time', 'Set', 'PulseSpeed', 'PulseRampTime', 'PulseDirection', 'PulseNumber']:
    #         df[f"{col}_gain"] = df[col] * df["gain_factor"]
    return df

def abs_mean_std_vect(x: np.ndarray):
    x_abs = np.abs(x)
    return np.mean(x_abs, axis=-1), np.std(x_abs, axis=-1)

def std_vect(x: np.ndarray):
    return np.std(x, axis=-1)

def mean_vect(x: np.ndarray):
    return np.mean(x, axis=-1)

def min_max_vect(x: np.ndarray):
    return np.min(x, axis=-1), np.max(x, axis=-1)

def quantiles_vect(
    x: np.ndarray,
    qs: List[float],
    add_ptp: bool = False,
    add_iqr: bool = False,
    axis=-1,
) -> np.ndarray:
    """Get the quantiles of a 1D signal (vectorized)

    Parameters
    ----------
    x : np.ndarray
        The input 1D signal.
    qs : List[float]
        The quantiles to compute.
    add_ptp : bool, optional
        Whether to add the peak-to-peak value as the last quantile, by default False.
        Note that the first (i.e., 0 = min) and land (i.e., 1 = max) quantiles should
        be included in `qs` if `add_ptp` is True.
    add_iqr : bool, optional
        Whether to add the interquartile range as the last value, by default False.
        Note that the first (i.e., 0.25 = q1) and land (i.e., 0.75 = q3) quantiles
        should be included in `qs` if `add_iqr` is True.
        If both `add_ptp` and `add_iqr` are True, the interquartile range will be
        appended tot the values as final value (with the peak-to-peak value before it).

    Returns
    -------
    np.ndarray
        of shape: [n_rows, len(qs) + add_ptp]

    """
    q_values = np.quantile(x, qs, axis=axis)
    if add_ptp:
        # the first and last quantile are 0 (i.e., min) and 1 (i.e., max)
        assert qs[0] == 0 and qs[-1] == 1
        q_values = np.concatenate(
            [q_values, (q_values[1] - q_values[0])[None, :]], axis=0
        )
    if add_iqr:
        # the first and last quantile are 0.25 (i.e., q1) and 0.75 (i.e., q3)
        assert 0.25 in qs and 0.75 in qs
        q1 = q_values[qs.index(0.25)]
        q3 = q_values[qs.index(0.75)]
        q_values = np.concatenate([q_values, (q3 - q1)[None, :]], axis=0).astype(
            x.dtype
        )
    return q_values.T

f_abs_mean_std = FuncWrapper(
    func=abs_mean_std_vect,
    vectorized=True,
    output_names=['abs_mean', 'abs_std']
)

f_qs_vect = FuncWrapper(
    quantiles_vect,
    vectorized=True,
    axis=-1,
    qs=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
    add_ptp=True,
    add_iqr=True,
    output_names=[
        "min",
        "q_0.1",
        "q_0.25",
        "median",
        "q_0.75",
        "q_0.9",
        "max",
        "ptp",
        "iqr",
    ],
) 

last = FuncWrapper(func=lambda x: x[:, -1], 
                   vectorized=True, 
                   output_names=['last'])
first = FuncWrapper(func=lambda x: x[:, 0], 
                    vectorized=True, 
                    output_names=['first'])
diff = FuncWrapper(func=lambda x: np.diff(x, axis=-1)[:, -1].flatten(), 
                   vectorized=True, 
                   output_names=['diff'])
f_std = FuncWrapper(func=std_vect, 
                    vectorized=True, 
                    output_names=['std'])
f_mean = FuncWrapper(func=mean_vect, 
                     vectorized=True, 
                     output_names=['mean'])
# f_min_max = FuncWrapper(func=min_max_vect, vectorized=True, output_names=['min', 'max'])

f_skew_v = FuncWrapper(ss.skew, 
                       vectorized=True, 
                       output_names="skew", axis=-1)
f_kurt_v = FuncWrapper(ss.kurtosis, 
                       vectorized=True, 
                       output_names="kurt", axis=-1)

def get_shift_config(df, w: int, time_col, shift_type: str = "next"):
    w_cols = seq(df.columns).filter(lambda x: re.search(f"_w={w}$", x)).to_list()
    if shift_type.startswith("prev"):
        number = shift_type.lstrip("prev")
        number = 1 if len(number) == 0 else int(number)
        f_df = df.loc[:, w_cols].add_suffix(f"_prev{number}").copy()
        f_df[time_col] = df[time_col] + number * w
        return f_df
    if shift_type.startswith("next"):
        number = shift_type.lstrip("next")
        number = 1 if len(number) == 0 else int(number)
        f_df = df.loc[:, w_cols].add_suffix(f"_next{number}").copy()
        f_df[time_col] = df[time_col] - number * w
        return f_df
    else:
        raise ValueError(f"Unknown shift: {shift_type}")

def val_model_catboost(feat_df, cur_feature_set):
    x_train = feat_df[0][cur_feature_set]
    x_val = feat_df[1][cur_feature_set]
    y_train = feat_df[0]['slip']
    y_val = feat_df[1]['slip']
    # if early_stopping:
    #     # TODO: With early stopping (which eval set?)
    #     model = CatBoostRegressor(iterations=10, verbose=0, od_wait=50, od_type="Iter", random_state=4, thread_count=6)
    # else:
    # Without early stopping
    model = CatBoostRegressor(
        iterations=50,
        learning_rate=0.07,
        depth=6,
        loss_function='RMSE',
        use_best_model=False,      # Disable best model selection
        verbose=0,
        task_type="CPU",
        random_seed=42,
        bootstrap_type="No",       # Deterministic training
        thread_count=6,            # Disable multithreading randomness
        random_strength=0         # Disable randomness in feature selection
    )
    model = model.fit(x_train, y_train, eval_set=(x_val, y_val))
    pred = model.predict(x_val)
    return mean_squared_error(pred, y_val)


def val_model_linear(feat_df, cur_feature_set):
    x_train = feat_df[0][cur_feature_set]
    x_val = feat_df[1][cur_feature_set]
    y_train = feat_df[0]['slip']
    y_val = feat_df[1]['slip']
    model = Pipeline([
        ('scaling', StandardScaler()),
        ("regressor", Lasso()),
    ])
    model.fit(x_train, y=y_train, regressor__sample_weight=y_train.abs())
    pred = model.predict(x_val)
    return mean_squared_error(pred, y_val)


def select_features(feature_names, feat_df, score_improvement_threshold=0.00025):
    selected_features = []
    scores = []
    improvement = True
    score = float('inf')

    features = set(feature_names)

    prev_score = score
    best_score = score
    while improvement:
        best_feature, improvement = None, False
        best_score = float('inf')

        # Create a closure
        def calc_score(feature):
            cur_feature_set = sorted(list(set(selected_features).union({feature})))
            try:
                # return val_model_linear(feat_df, cur_feature_set)
                return val_model_catboost(feat_df, cur_feature_set)
            except:
                print(f"Error with feature {feature}")
                return float('inf')

        features_ = sorted(list(features))

        with multiprocess.Pool(processes=6) as pool:
            results = list(tqdm(pool.imap(calc_score, features_), total=len(features_)))

        for score, feature in zip(results, features_):
            if score < best_score and score < prev_score - score_improvement_threshold:
                best_score = score
                best_feature = feature
                improvement = True
                
        if best_feature is not None:
            prev_score = best_score
            selected_features.append(best_feature)
            scores.append(best_score)
            features = features.difference({best_feature})
            
        print(improvement, best_score, best_feature)

    return selected_features, scores

if __name__ == "__main__":
    # Load files
    df_train = pd.read_parquet("./../../data/processed/processed_train.parquet")
    df_validation = pd.read_parquet("./../../data/processed/processed_validation.parquet")
    df_test = pd.read_parquet("./../../data/processed/processed_test.parquet")
    TARGET_COL = "slip"
    TIME_COL = "time_idx"
    GROUP_COL = "File"

    ## Compute the target variable
    RADIUS = 0.074
    df_train[TARGET_COL] = (
        -df_train["Encoders.Encoder1.Speed"] * 2 * np.pi * RADIUS / 2
        - df_train["Motors.Traction1.Speed"]
    )
    df_validation[TARGET_COL] = (
        -df_validation["Encoders.Encoder1.Speed"] * 2 * np.pi * RADIUS / 2
        - df_validation["Motors.Traction1.Speed"]
    )
    df_test[TARGET_COL] = (
        -df_test["Encoders.Encoder1.Speed"] * 2 * np.pi * RADIUS / 2
        - df_test["Motors.Traction1.Speed"]
    )

    # Set the time index column its name
    df_train.index.name = TIME_COL
    df_validation.index.name = TIME_COL
    df_test.index.name = TIME_COL

    # make the grouping column a category
    df_train[GROUP_COL] = df_train[GROUP_COL].astype('category')
    df_validation[GROUP_COL] = df_validation[GROUP_COL].astype('category')
    df_test[GROUP_COL] = df_test[GROUP_COL].astype('category')

    df_train = process_signals(df_train)
    df_validation = process_signals(df_validation)
    df_test = process_signals(df_test)

    s_names = [
        #     # The torque sensors
        #     # "Motors.Roller1.Torque",
        # "Motors.Dancer1.Torque",
        # "Motors.Dancer1.Position",
        #     # "Motors.Dancer1.Speed",
        #     # "Motors.Roller1.WebSpeedSimple",
        #     "Motors.Roller1.SlipSimple",
        # "Motors.Traction1.Torque",
        "Traction1_diff(speed)",
        # "Motors.Traction1.Speed",
        # "Motors.Traction1.Setpoint_diff",
        # "Motors.Traction1.Position_diff",
        # "Motors.Traction1.Speed",  # is good
        # "Motors.Traction1.Setpoint", # setpoint and speed are 99.99% correlated
        # "Motors.Traction2.Torque",
        # "Motors.Traction2.Speed",  # is good
        #     # "Motors.Accumulator.Torque",
        #     # and the load cell sensor
        # "AnalogSensors.LoadCell1",
        # "AnalogSensors.LoadCell2",
        #     # The engineered signals
        #     # "Dancer1_diff(delta(torque, speed))",
        # "Dancer1_delta(torque, speed)",  # TODO -> has speed incorporated
        "Traction1_diff(delta(torque, speed))",  # is good
        # "Traction2_diff(delta(torque, speed))", # is good
        "delta_traction_speed",
        # "Traction1_delta(torque,diff(speed))",
        "LoadCell1_diff",
        "LoadCell2_diff",
        "diff(delta(LoadCell1, LoadCell2))",
        "delta(LoadCell1, LoadCell2)",
    ]

    # Append s_names with all gain factor versions of the current signals in s_names
    # s_names += [f"{col}_gain" for col in s_names]
    # Add the gain factor itself as a signal to s_names
    # s_names += ["gain_factor"]
    # Remove all signals that don't contain "gain"
    # s_names = [col for col in s_names if "gain" in col]

    granular_s_names = [
        # "Motors.Traction1.Setpoint_diff",
        # "Motors.Traction1.Position_diff",
        # "Motors.Traction2.Setpoint_diff",
        # "Motors.Traction2.Position_diff",
    ]


    diff_s_names = [
        "Motors.Dancer1.Position",
        "Motors.Dancer1.Speed",
        # "Motors.Accumulator.Position",
        # "Motors.Accumulator.Speed",
        "Motors.Roller1.Position",
        "Motors.Roller1.Speed",
        "Motors.Traction1.Position",
        "Motors.Traction1.Setpoint",
        "Motors.Traction2.Position",
        "Motors.Traction2.Setpoint",
        # "AnalogSensors.LoadCell1",
    ]

    # diff_s_names += [f"{col}_gain" for col in diff_s_names]
    # Remove all signals that don't contain "gain"
    # diff_s_names = [col for col in diff_s_names if "gain" in col]

    # s_names = list(set(df_train.columns).difference({'File', 'Slip'}))

    STRIDE = 1
    fc = FeatureCollection(
        feature_descriptors=[
            MultipleFeatureDescriptors(
                functions=[f_std, f_mean],  # , f_skew_v, f_kurt_v],
                series_names=s_names,
                windows=[4, 8, 16, 32, 64],
                strides=STRIDE,
            ),
            # MultipleFeatureDescriptors(
            #     functions=[f_qs_vect], # f_skew_v, f_kurt_v],
            #     series_names=s_names,
            #     windows=[16, 32, 64],
            #     strides=STRIDE,
            # ),
            MultipleFeatureDescriptors(
                functions=[last, first, diff],
                series_names=s_names + granular_s_names,
                windows=[2],
                strides=STRIDE,
            ),
            MultipleFeatureDescriptors(
                functions=[diff], series_names=diff_s_names, windows=[2], strides=STRIDE
            ),
            MultipleFeatureDescriptors(
                functions=[first],
                series_names=["UserInput.WebDirection"],
                windows=[1],
                strides=STRIDE,
            ),
        ]
    )

    df_feat_train_ = fc.calculate(df_train, show_progress=True, n_jobs=None, return_df=True)
    df_feat_train_ = df_feat_train_.reset_index(drop=False)
    df_feat_train_ = df_feat_train_.dropna()  # we drop the nan values so that we can use sklearn later on

    df_feat_val_ = fc.calculate(df_validation, show_progress=True, n_jobs=None, return_df=True)
    df_feat_val_ = df_feat_val_.reset_index(drop=False)
    df_feat_val_ = df_feat_val_.dropna()  # we drop the nan values so that we can use sklearn later on

    df_feat_test_ = fc.calculate(df_test, show_progress=True, n_jobs=None, return_df=True)
    df_feat_test_ = df_feat_test_.reset_index(drop=False)
    df_feat_test_ = df_feat_test_.dropna()

    df_feat_train = df_feat_train_
    df_feat_val = df_feat_val_
    df_feat_test = df_feat_test_

    for w_size, shift in [
        (2, "next"),
        (2, "next2"),
        (2, "next4"),
        (2, "next8"),
        # (2, "next16"),

        (4, "prev"),
        (4, "prev2"),
        (4, "next"),
        (8, "prev"),
        (8, "prev2"),
        (8, "next2"),

        (2, "prev"),
        (2, "prev2"),
        (2, "prev4"),
        (2, "prev8"),
        (2, "prev12"),
        (2, "prev16"),
        # 
        # (4, "prev"),
        # (8, "prev"),
        # (16, "prev"),
        # (4, "next"),
        # (8, "next"),
        # (16, "next"),
    ]:  # , (4, "next")]: #, (8, "next"), (16, "next")]:
        df_feat_train = df_feat_train.merge(
            get_shift_config(df_feat_train_, w_size, TIME_COL, shift), on=TIME_COL
        )
        df_feat_val = df_feat_val.merge(
            get_shift_config(df_feat_val_, w_size, TIME_COL, shift), on=TIME_COL
        )
        df_feat_test = df_feat_test.merge(
            get_shift_config(df_feat_test, w_size, TIME_COL, shift), on=TIME_COL
        )

    df_feat_train_m = pd.merge(
        df_feat_train.set_index(TIME_COL),
        df_train[[GROUP_COL, TARGET_COL] + ['Set', 'PulseSpeed', 'PulseRampTime', 'PulseDirection', 'PulseNumber']],
        left_index=True,
        right_index=True,
        how="left",
    )
    df_feat_val_m = pd.merge(
        df_feat_val.set_index(TIME_COL),
        df_validation[[GROUP_COL, TARGET_COL] + ['Set', 'PulseSpeed', 'PulseRampTime', 'PulseDirection', 'PulseNumber']],
        left_index=True,
        right_index=True,
        how="left",
    )

    feat_cols = list(set(df_feat_train_m.columns) - {TIME_COL, GROUP_COL, TARGET_COL} - set(['Set', 'PulseSpeed', 'PulseRampTime', 'PulseDirection', 'PulseNumber']))
    print(f"Number of features before selection: {len(feat_cols)}")
    print('Selecting features')
    # Select features
    selected_features, scores = select_features(feat_cols, (df_feat_train_m, df_feat_val_m), score_improvement_threshold=0)
    # Lists to dataframe
    sel_df = pd.DataFrame({'selected_features':selected_features, 'scores': scores})
    sel_df.to_csv('./../../data/features/selected_features_catboost.csv')