from collections import Counter
from scipy.special import erfinv
from sklearn.preprocessing import StandardScaler
from optimization import lightgbm_optimizer
from op

import time

result = None

x, y = clinical.values[:,1:], clinical.values[:, 0]

kfold = StratifiedKFold(10, shuffle=True, random_state=185)

aucs, distances = [], []

for i, (train_index, valid_index) in enumerate(kfold.split(x, y)):
    
    overall_start = time.time()
    
    print('Fold #{}'.format(i + 1))
    
    #
    # Split train & valid
    #
    start = time.time()
    
    response_train = clinical.iloc[train_index, [0]].values
    response_valid = clinical.iloc[valid_index, [0]].values
    
    clinical_scaler_minmax = MinMaxScaler()
    
    clinical_train = clinical.iloc[train_index, 1:]
    clinical_train = pd.DataFrame(np.maximum(0, np.minimum(1, clinical_scaler_minmax.fit_transform(clinical_train.values))), 
                                  columns=clinical_train.columns, index=clinical_train.index).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    clinical_valid = clinical.iloc[valid_index, 1:]
    clinical_valid = pd.DataFrame(np.maximum(0, np.minimum(1, clinical_scaler_minmax.transform(clinical_valid.values))), 
                                  columns=clinical_valid.columns, index=clinical_valid.index).replace([np.inf, -np.inf], np.nan).fillna(0)
                                  
    genefpkm_scaler_robust = RobustScaler()
    genefpkm_scaler_minmax = MinMaxScaler()
    
    genefpkm_train = genefpkm.iloc[train_index, :]
    genefpkm_train = pd.DataFrame(np.maximum(0, np.minimum(1, genefpkm_scaler_minmax.fit_transform(
        genefpkm_scaler_robust.fit_transform(genefpkm_train.values)))), 
                                  columns=genefpkm_train.columns, index=genefpkm_train.index).replace(
        [np.inf, -np.inf], np.nan).fillna(0)
    
    genefpkm_valid = genefpkm.iloc[valid_index, :]
    genefpkm_valid = pd.DataFrame(np.maximum(0, np.minimum(1, genefpkm_scaler_minmax.transform(
        genefpkm_scaler_robust.transform(genefpkm_valid.values)))), 
                                  columns=genefpkm_valid.columns, index=genefpkm_valid.index).replace(
        [np.inf, -np.inf], np.nan).fillna(0)
    
    print('Data normalization compledted in {} seconds'.format(time.time() - start))
    
    #
    # Select gene expressions
    #
    start = time.time()
    
    print('Selecting gene expressions and features')
    
    if os.path.isfile('output/brfl/selected_genes_fold_{}.pkl'.format(i)):
        
        with open('output/brfl/selected_genes_fold_{}.pkl'.format(i), 'rb') as file:
            selected_genes = sorted(list(pickle.load(file)))
            
        with open('output/brfl/selected_feats_fold_{}.pkl'.format(i), 'rb') as file:
            selected_feats = sorted(list(pickle.load(file)))
    
    else:
        
        selected_genes = sorted(list(select_genes(genefpkm_train, response_train[:,0], threshold=0.03)))
        
        selected_feats = sorted(list(select_genes(clinical_train, response_train[:,0], threshold=0.05)))
        
        with open('output/brfl/selected_genes_fold_{}.pkl'.format(i), 'wb') as file:
            pickle.dump(selected_genes, file)
            
        with open('output/brfl/selected_feats_fold_{}.pkl'.format(i), 'wb') as file:
            pickle.dump(selected_feats, file)

    print('Selecting {} gene expressions of {}'.format(len(selected_genes), genefpkm.shape[1]))
    print('Selecting {} clinical variables of {}'.format(len(selected_feats), clinical.shape[1]))
    
    genefpkm_train = genefpkm_train.loc[:,selected_genes]
    clinical_train = clinical_train.loc[:,selected_feats]
    
    genefpkm_valid = genefpkm_valid.loc[:,selected_genes]
    clinical_valid = clinical_valid.loc[:,selected_feats]
    
    print('Features selected in {} seconds'.format(time.time() - start))

    #
    # Train & Test distances
    #
    dists = []
    
    for row_train in clinical_train.join(genefpkm_train, how='inner').values:
        for row_valid in clinical_valid.join(genefpkm_valid, how='inner').values:
            dists.append(np.linalg.norm(row_train-row_valid))
            
    print("Distance between train and test data sets: ", "MEAN: ", np.mean(dists), " STDVED: ", np.std(dists))
    
    distances.append(np.mean(dists))
    
    #
    # Genetic Profiling
    #
    start = time.time()
    
    print('Computing genetic profling')
    
    if os.path.isfile('output/brfl/kmeans_genetic_profiling_fold_{}.pkl'.format(i)):
        
        with open('output/brfl/kmeans_genetic_profiling_fold_{}.pkl'.format(i), 'rb') as file:
            genetic_profiling = pickle.load(file)
        
    else:
        
        genetic_profiling = GeneticProfiling(random_state=10)

        genetic_profiling.fit(genefpkm_train)
        
        with open('output/brfl/kmeans_genetic_profiling_fold_{}.pkl'.format(i), 'wb') as file:
            pickle.dump(genetic_profiling, file)
        
    profiling_train = to_data_frame(genetic_profiling.predict(genefpkm_train).reshape((-1, 1)), prefix='PV', index=genefpkm_train.index)
    clinical_train = pd.concat([clinical_train, profiling_train], axis=1)
    
    profiling_valid = to_data_frame(genetic_profiling.predict(genefpkm_valid).reshape((-1, 1)), prefix='PV', index=genefpkm_valid.index)
    clinical_valid = pd.concat([clinical_valid, profiling_valid], axis=1)    
    
    print('Genetic profiling computed in {} seconds'.format(time.time() - start))
    
    #
    # Gene Clustering
    #
    start = time.time()
    
    print('Computing genetic clustering')
    
    if os.path.isfile('output/brfl/kmeans_genetic_clustering_fold_{}.pkl'.format(i)):
        
        with open('output/brfl/kmeans_genetic_clustering_fold_{}.pkl'.format(i), 'rb') as file:
            genetic_clustering = pickle.load(file)
        
    else:
        
        genetic_clustering = GeneticClustering(random_state=10, verbose=0, early_stopping_rounds=10)

        genetic_clustering.fit(genefpkm_train)
        
        with open('output/brfl/kmeans_genetic_clustering_fold_{}.pkl'.format(i), 'wb') as file:
            pickle.dump(genetic_clustering, file)
    
    gene_cluster_train = to_data_frame(genetic_clustering.transform(genefpkm_train), prefix='GC', index=genefpkm_train.index)
    clinical_train = pd.concat([clinical_train, gene_cluster_train], axis=1)
    
    gene_cluster_valid = to_data_frame(genetic_clustering.transform(genefpkm_valid), prefix='GC', index=genefpkm_valid.index)
    clinical_valid = pd.concat([clinical_valid, gene_cluster_valid], axis=1)
    
    print('Genetic clustering completed in {} seconds'.format(time.time() - start))
    
    #
    # Denoising Autoencoder
    #
    start = time.time()
    
    print('Denoising autoencoder')
    
    dae_prefix = 'data_augmentation_adadelta_fold'
    
    dae = DenoisingAutoencoder(model_name='{}_{}'.format(dae_prefix, i), summaries_dir='output/brfl/deep_models/', verbose=1);
    
    if not os.path.exists('output/brfl/deep_models/{0}_{1}/graph/{0}_{1}.meta'.format(dae_prefix, i)):
        
        dae.build(n_inputs=genefpkm_train.shape[1], 
                  encoder_units=(int(genefpkm_train.shape[1] * .5), int(genefpkm_train.shape[1] * .3), int(genefpkm_train.shape[1] * .2)), 
                  decoder_units=(int(genefpkm_train.shape[1] * .3), int(genefpkm_train.shape[1] * .5)), 
                  encoder_activation_function='relu', decoder_activation_function='identity', l2_scale=0.01);
        
        dae.fit(genefpkm_train.values, steps=10000, optimizer='adadelta', loss='mse', learning_rate=1e-2);#, keep_probability=0.5)
        
    dae.load('output/brfl/deep_models/{0}_{1}/graph/{0}_{1}'.format(dae_prefix, i));
    
    dda_scaler = MinMaxScaler()
    
    dda_train = dae.predict(genefpkm_train.values)
    dda_train = pd.DataFrame(dda_scaler.fit_transform(dda_train), index=genefpkm_train.index)
    dda_train.columns = [col + '_DDA' for col in genefpkm_train.columns]
    
    dda_valid = dae.predict(genefpkm_valid.values)
    dda_valid = pd.DataFrame(dda_scaler.transform(dda_valid), index=genefpkm_valid.index)
    dda_valid.columns = [col + '_DDA' for col in genefpkm_valid.columns]
    
    dae.close()
    
    del dae
    
    x_train = clinical_train.join(genefpkm_train, how='inner').join(dda_train, how='inner')
    x_valid = clinical_valid.join(genefpkm_valid, how='inner').join(dda_valid, how='inner')
    
    print('Denoising autoencoder fitted in {} seconds'.format(time.time() - start))
    '''
    #
    # Dense
    #
    start = time.time()

    print('Multiple Dense Models')
    
    md_prefix = 'dense_adagrad_geneexp_3L256N_lr=1e-1_l2=1e-8_dout=5e-1_fold'
    
    multiple_dense = Dense(model_name='{}_{}'.format(md_prefix, i), summaries_dir='output/brfl/deep_models/')
    
    if not os.path.exists('output/brfl/deep_models/{0}_{1}/{0}_{1}.meta'.format(md_prefix, i)):
        
        multiple_dense.build(n_input_features=x_train.shape[1], 
                             n_outputs=1, 
                             abstraction_activation_functions=('sigmoid', 'tanh', 'relu'),
                             n_hidden_layers=3, n_hidden_nodes=256, 
                             keep_probability=0.5,
                             optimizer_algorithms=('adagrad', 'adagrad', 'adagrad'), 
                             cost_function='logloss', 
                             add_summaries=True,
                             batch_normalization=True, l2_regularizer=1e-8)
        
        multiple_dense.fit(x_train.values, response_train, x_valid, 
                           response_valid, learning_rate=1e-1, 
                           steps=10000, batch_size=x_train.shape[0], shuffle=True)
    
    multiple_dense.load('output/brfl/deep_models/{0}_{1}/{0}_{1}'.format(md_prefix, i))

    md_train = np.mean(multiple_dense.predict(x_train.values).T[0], axis=1).reshape((-1, 1))
    md_train = pd.DataFrame(md_train, columns=['MD{}'.format(ll + 1) for ll in range(md_train.shape[1])], index=x_train.index)
    
    md_valid = np.mean(multiple_dense.predict(x_valid.values).T[0], axis=1).reshape((-1, 1))
    md_valid = pd.DataFrame(md_valid, columns=['MD{}'.format(ll + 1) for ll in range(md_valid.shape[1])], index=x_valid.index)
    
    multiple_dense.close()
    
    del multiple_dense
    
    print('Multiple MLPs fitted in {} seconds'.format(time.time() - start))
    
    #
    # Conv Dense
    #
    
    for ii in range(x_train_transformed.shape[2]):
                
        x_max, x_min = np.max(x_train_transformed[:, :, ii:, :, :]), np.min(x_train_transformed[:, :, ii, :, :])
    
        x_train_transformed[:, :, ii, :, :] = (x_train_transformed[:, :, ii, :, :] - x_min) / (x_max - x_min)

        x_valid_transformed[:, :, ii, :, :] = (x_valid_transformed[:, :, ii, :, :] - x_min) / (x_max - x_min)
    
    cd_prefix = 'convdense_3h256ndout=5e-1_lr=1e-1_l2=0_adadelta_logloss_minmax_normnode'
    
    conv_dense = ConvDense(model_name='{}_fold_{}'.format(cd_prefix, i), summaries_dir='output/brfl/deep_models/', verbose=1)
    
    if not os.path.exists('output/brfl/deep_models/{0}_fold_{1}/graph/{0}_fold_{1}.meta'.format(cd_prefix, i)):
    
        conv_dense.build(n_models=3, n_neurons_per_layer=256, n_layers=3, 
                         n_outputs=1, optimizer_algorithm='adadelta', keep_probability=0.5, loss='logloss')

        conv_dense.fit(x_train_transformed, response_train, x_valid_transformed, response_valid, 
                       batch_size=x_train.shape[0], steps=500, learning_rate=1e-1)    
    
    conv_dense.load('output/brfl/deep_models/{0}_fold_{1}/graph/{0}_fold_{1}'.format(cd_prefix, i))
    
    x_train = conv_dense.transform(x_train_transformed)
    
    x_valid = conv_dense.transform(x_valid_transformed)
    
    conv_dense.close()
    
    del conv_dense
    
    #
    #
    #
    start = time.time()
    
    print('Optimizing')
    
    x_train = x_train.join(md_train, how='inner')
    x_valid = x_valid.join(md_valid, how='inner')
    '''
    file_name = 'output/brfl/optimization_lgbm_fold_{}.pkl'.format(i)
    
    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            opt = pickle.load(file)
    else:
        opt = lightgbm_optimizer(x_train.values, response_train, space, n_calls=100).x
        
        with open(file_name, 'wb') as file:
            pickle.dump(opt, file)
    
    print('Optimization completed in {} seconds'.format(time.time() - start))
    
    #
    # LightGBM
    #
    start = time.time()
    
    print('Training')
    
    params = {
        'learning_rate': opt[0],
        'num_leaves': opt[1],
        'max_depth': opt[2],
        'scale_pos_weight': opt[3],
        'min_child_weight': opt[4],
        'colsample_bytree': opt[5],
        'min_split_gain': opt[6],
        'min_child_samples': opt[7],
        'subsample': opt[8],
        
        'subsample_for_bin': 2,
        'objective':'binary',
        'metric':'auc',
        'eval_metric':'auc',
        'is_unbalance':False,
        'nthread':24,          
        'verbose': -1}
    
    lgb_train = lgb.Dataset(x_train.values, list(response_train.reshape((-1,))))
    lgb_valid = lgb.Dataset(x_valid.values, list(response_valid.reshape((-1,))))

    gbm = lgb.train(params, lgb_train, valid_sets=lgb_valid, num_boost_round=100000, 
                    early_stopping_rounds=100, verbose_eval=True)

    print('LightGBM fitted in {} seconds'.format(time.time() - start))
    
    #
    #
    #
    y_ = gbm.predict(x_valid.values, num_iteration=gbm.best_iteration, verbose_eval=False)
    
    auc = roc_auc_score(response_valid, y_)

    print(i + 1, auc)
    
    aucs.append(auc)
    
    #
    # 
    #    
    print('Fold completed in {} seconds\n\n'.format(time.time() - overall_start))
