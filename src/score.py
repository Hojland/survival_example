def main():
    logger = general_utils.get_logger("/app/printyboi.log")
    mariadb_engine = sql_utils.create_engine(**settings.MARIADB_CONFIG)
    mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    cluster = LocalCUDACluster()
    client = Client(cluster)
    client.run(cudf.set_allocator, "managed")
    for main_product in settings.MODEL_PRODUCTS:

        logger.info("loading data and building dataset")
        df = SurvivalDataset(
            datasets=["online_data"],
            main_product=main_product,
            survival_objective=SURVIVAL_OBJECTIVE,
        )()
        df = df.compute()

        df = df.loc[~(df[settings.ONLINE_COLUMNS + settings.CAMPAIGN_COLUMNS].sum(axis=1) == 0)]

        # TODO
        # figure out conditional_after implementation

        X = df.loc[df.label_upper_bound == cp.inf].drop(
            columns=["label_lower_bound", "label_upper_bound"]
        )

        logger.info("loading the model")
        xgboo = XgboostGPU()
        xgboo.load_model(MODEL_NAME.format(main_product), MODEL_STAGE)

        logger.info("calculate shap values")
        xgboo.explain(X)
        shap = xgboo.shap_values
        shap_argsort = np.argsort(np.abs(shap))[:, ::-1]

        logger.info("get predictions and output df")
        out_df = X.reset_index(drop=False)[["index"]].rename({"index": "linkitid"}, axis=1).copy()

        out_df["prediction"] = xgboo.predict(X).values.get()
        X_array = X.values.get()  # X.iloc is very slow

        # top 5 important features
        for i in range(5):
            out_df[f"feature_{i+1}"] = X.columns[shap_argsort[:, i]]
            out_df[f"value_{i+1}"] = [shap[n, col] for n, col in enumerate(shap_argsort[:, i])]
            out_df[f"feature_value_{i+1}"] = [
                X_array[n, col] for n, col in enumerate(shap_argsort[:, i])
            ]

        dtype_trans_dict = sql_utils.get_dtype_trans(out_df, str_len=500)

        out_df = out_df.to_pandas()

        out_df.to_sql(
            SQL_TABLE_NAME.format(main_product),
            schema=SQL_SCHEMA,
            con=mariadb_engine,
            if_exists="replace",
            index=False,
            dtype=dtype_trans_dict,
        )


if __name__ == "__main__":
    main()