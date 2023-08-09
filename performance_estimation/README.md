# Brief guidance on how to run the codes
0. Prepare data via data_processing/build_basic.csv and build_temporal_feature.csv
1. Run DAE_main.py to train DAE, save the model and DAE-augmented features
2. Run DAE_Xgboost.py to train Xgboost based on DAE results
3. Run DAEDAE.py to transfer the learned DAE to unseen applications via adversarial loss