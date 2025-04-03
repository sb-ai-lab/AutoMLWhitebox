import time

from sklearn.metrics import roc_auc_score

from autowoe import ReportDeco, AutoWoE


def test_dates_and_stat_model(train_data, test_data, test_target):

    train = train_data
    test = test_data

    test["target"] = test_target.values

    num_col = list(filter(lambda x: "numb" in x, train.columns))
    num_feature_type = {x: "real" for x in num_col}

    date_col = list(filter(lambda x: "datetime" in x, train.columns))
    date_feature_type = {x: (None, ("d", "wd")) for x in date_col}

    features_type = dict(**num_feature_type, **date_feature_type)
    # подробно параметры описаны в Example_1
    auto_woe = AutoWoE(
        monotonic=True, max_bin_count=4, oof_woe=False, regularized_refit=False, p_val=0.05, debug=False, verbose=0
    )
    auto_woe = ReportDeco(auto_woe)

    start_fit_time = time.time()
    auto_woe.fit(
        train[num_col + date_col + ["target"]],
        target_name="target",
        features_type=features_type,
    )

    assert time.time() - start_fit_time < 50, f"Fit time is {time.time() - start_fit_time}, it's more than 50"

    start_pred_time = time.time()
    pred = auto_woe.predict_proba(test)

    assert time.time() - start_pred_time < 5, f"Predict time is {time.time() - start_pred_time}, it's more than 5"

    score = roc_auc_score(test["target"], pred)

    assert score > 0.78

    report_params = {
        "automl_date_column": "report_month",  # колонка с датой в формате params['datetimeFormat']
        "output_path": "./AUTOWOE_REPORT_2",  # папка, куда сгенерится отчет и сложатся нужные файлы
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

    # import shutil

    # shutil.rmtree("AUTOWOE_REPORT_2")
