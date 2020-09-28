optimal_threshold = optimize_threshold(response_train, y_train_hat)

    if optimal_threshold is None:
        optimal_threshold = np.median(y_train_hat)
    y_valid_optimized = [int(y_ > optimal_threshold) for y_ in y_valid_hat]

    tn, fp, fn, tp = confusion_matrix(response_valid, y_valid_optimized).ravel()

    metrics = classification_metrics(tn, fp, fn, tp)
    ks = ks_score(response_valid, y_valid_hat)

    auc_train = roc_auc_score(response_train, y_train_hat)
    auc_valid = roc_auc_score(response_valid, y_valid_hat)

    overall_time = time.time() - overall_start

    print(fold, n_features, auc_train, auc_valid)