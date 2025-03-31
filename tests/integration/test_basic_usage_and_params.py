import time
import numpy as np

from sklearn.metrics import roc_auc_score

from autowoe import ReportDeco, AutoWoE


def test_basic_usage_and_params(train_data, test_data, test_target):

    train = train_data

    train = train.iloc[:, 50:100]

    num_col = list(filter(lambda x: "numb" in x, train.columns))
    num_feature_type = {x: "real" for x in num_col}

    date_col = filter(lambda x: "datetime" in x, train.columns)
    for col in date_col:
        train[col + "_year"] = train[col].map(lambda x: x.year)
        train[col + "_weekday"] = train[col].map(lambda x: x.weekday())
        train[col + "_month"] = train[col].map(lambda x: x.month)

    test = test_data

    date_col = filter(lambda x: "datetime" in x, test.columns)
    for col in date_col:
        test[col + "_year"] = test[col].map(lambda x: x.year)
        test[col + "_weekday"] = test[col].map(lambda x: x.weekday())
        test[col + "_month"] = test[col].map(lambda x: x.month)

    test["target"] = test_target.values

    cat_col = list(filter(lambda x: "str" in x, train.columns))
    cat_feature_type = {x: "cat" for x in cat_col}

    year_col = list(filter(lambda x: "_year" in x, train.columns))
    year_feature_type = {x: "cat" for x in year_col}

    weekday_col = list(filter(lambda x: "_weekday" in x, train.columns))
    weekday_feature_type = {x: "cat" for x in weekday_col}

    month_col = list(filter(lambda x: "_month" in x, train.columns))
    month_feature_type = {x: "cat" for x in month_col}

    features = cat_col + year_col + weekday_col + month_col + num_col

    features_type = dict(
        **num_feature_type, **cat_feature_type, **year_feature_type, **weekday_feature_type, **month_feature_type
    )

    features_monotone_constraints = {"number_74": "auto", "number_83": "auto"}

    max_bin_count = {"number_47": 3, "number_51": 2}

    auto_woe = AutoWoE(
        task="BIN",
        interpreted_model=True,
        monotonic=False,
        max_bin_count=5,
        select_type=None,
        pearson_th=0.9,
        auc_th=0.505,
        vif_th=10.0,
        imp_th=0,
        th_const=32,
        force_single_split=True,
        th_nan=0.01,
        th_cat=0.005,
        woe_diff_th=0.01,
        min_bin_size=0.01,
        min_bin_mults=(2, 4),
        min_gains_to_split=(0.0, 0.5, 1.0),
        auc_tol=1e-4,
        cat_alpha=100,
        cat_merge_to="to_woe_0",
        nan_merge_to="to_woe_0",
        oof_woe=True,
        n_folds=6,
        n_jobs=4,
        l1_grid_size=20,
        l1_exp_scale=6,
        imp_type="feature_imp",
        regularized_refit=False,
        p_val=0.05,
        debug=False,
        verbose=0,
    )

    auto_woe = ReportDeco(auto_woe)

    start_fit_time = time.time()
    auto_woe.fit(
        train[features + ["target"]],
        target_name="target",
        features_type=features_type,
        group_kf=None,
        max_bin_count=max_bin_count,
        features_monotone_constraints=features_monotone_constraints,
        validation=test,
    )

    assert time.time() - start_fit_time < 25, f"Fit time is {time.time() - start_fit_time}, it's more than 25"

    start_predict_time = time.time()
    pred = auto_woe.predict_proba(test)
    assert (
        time.time() - start_predict_time < 3.5
    ), f"Predict time is {time.time() - start_predict_time}, it's more than 3.5"

    score_1 = roc_auc_score(test["target"], pred)

    assert score_1 > 0.76

    # assert np.isclose(score_1, 0.7791178), f"Real score is {score_1}"

    pred = auto_woe.predict_proba(test[["number_72"]], report=False)
    score_2 = roc_auc_score(test["target"], pred)

    assert np.isclose(score_1, score_2), f"Scores {score_1} and {score_2} musts be equal"

    report_params = {
        "automl_date_column": "report_month",  # колонка с датой в формате params['datetimeFormat']
        "output_path": "./AUTOWOE_REPORT_1",  # папка, куда сгенерится отчет и сложатся нужные файлы
        "report_name": "___НАЗВАНИЕ ОТЧЕТА___",
        "report_version_id": 1,
        "city": "Воронеж",
        "model_aim": "___ЦЕЛЬ ПОСТРОЕНИЯ МОДЕЛИ___",
        "model_name": "___НАЗВАНИЕ МОДЕЛИ___",
        "zakazchik": "___ЗАКАЗЧИК___",
        "high_level_department": "___ПОДРАЗДЕЛЕНИЕ___",
        "ds_name": "___РАЗРАБОТЧИК МОДЕЛИ___",
        "target_descr": "___ОПИСАНИЕ ЦЕЛЕВОГО СОБЫТИЯ___",
        "non_target_descr": "___ОПИСАНИЕ НЕЦЕЛЕВОГО СОБЫТИЯ___",
    }

    auto_woe.generate_report(report_params)
