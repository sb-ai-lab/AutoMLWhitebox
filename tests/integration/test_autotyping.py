import time

from sklearn.metrics import roc_auc_score

from autowoe import ReportDeco, AutoWoE


def test_autotyping(cat_data):

    data = cat_data

    train = data.iloc[:14000, :]
    test = data.iloc[14000:, :]

    # подробно параметры описаны в Example_1
    auto_woe = AutoWoE(
        monotonic=False,
        max_bin_count=5,
        oof_woe=True,
        regularized_refit=True,
        p_val=0.05,
        debug=False,
        verbose=0,
        cat_merge_to="to_maxp",
        nan_merge_to="to_maxp",
    )
    auto_woe = ReportDeco(auto_woe)

    autowoe_fit_params = {
        "train": train,
        "target_name": "isFraud",
    }
    start_fit_time = time.time()
    auto_woe.fit(**autowoe_fit_params)

    assert time.time() - start_fit_time < 60, f"Fit time is {time.time() - start_fit_time}, it's more than 60 seconds"

    start_pred_time = time.time()
    pred = auto_woe.predict_proba(test)

    assert (
        time.time() - start_pred_time < 5
    ), f"Prediction time is {time.time() - start_pred_time}, it's more than 5 seconds"

    score = roc_auc_score(test[autowoe_fit_params["target_name"]], pred)

    assert score > 0.8

    values = {}
    for value in auto_woe.private_features_type.values():
        if value not in values:
            values[value] = 0
        values[value] += 1

    assert (
        values["cat"] == 12 and values["real"] == 61
    ), f"There're should be 12 cat and 61 reals, but we have {values['cat']} cats and {values['real']} reals"

    report_params = {
        "automl_date_column": "report_month",  # колонка с датой в формате params['datetimeFormat']
        "output_path": "./AUTOWOE_REPORT_3",  # папка, куда сгенерится отчет и сложатся нужные файлы
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
