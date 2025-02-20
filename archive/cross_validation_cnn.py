# reg_param_lst = np.concatenate([np.zeros(1), np.logspace(-3, 3, 7)])
# losses_df = pd.DataFrame(columns=["alpha", "split", "val_loss"])
# cv_splits = StratifiedKFold(n_splits=5)


# # use 5-fold cross-validation on the train + validation sets mixed together stratifying by binary resistance phenotype
    # # use the same 5 splits for all regularization parameters to reduce variance
    # for split, (cv_train_idx, cv_val_idx) in enumerate(cv_splits.split(df_train_val.index, df_train_val["Binary"])):
        
    #     split_losses = []
            
    #     cv_train_generator = MtbGeneDataset(
    #                                     drug,
    #                                     df_train_val,
    #                                     os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'),
    #                                     protein_embeddings_PC_fName,
    #                                     seq_data_path=seq_data_path,
    #                                     binary=binary,
    #                                     cc=binary_thresh,
    #                                     tier1_loci=tier1_loci,
    #                                     tier2_loci=tier2_loci,
    #                                     embed_genes_list=embed_genes_list,
    #                                     data_subset=None,
    #                                     data_idx=cv_train_idx,
    #                                     shuffle_phenos=False,
    #                                     include_lineage=include_lineage,
    #                                     include_peptide_lengths=include_peptide_lengths,
    #                                     include_protein_embeddings=include_protein_embeddings,
    #                                     bounded_loss=bounded_loss,
    #                                     batch_size=BATCH_SIZE,
    #                                     shuffle_batches=True
    #     )
        
    #     cv_val_generator = MtbGeneDataset(
    #                                     drug,
    #                                     df_train_val,
    #                                     os.path.join(seq_data_path, 'pkl_sparse_train_val.npz'),
    #                                     protein_embeddings_PC_fName,
    #                                     seq_data_path=seq_data_path,
    #                                     binary=binary,
    #                                     cc=binary_thresh,
    #                                     tier1_loci=tier1_loci,
    #                                     tier2_loci=tier2_loci,
    #                                     embed_genes_list=embed_genes_list,
    #                                     data_subset=None,
    #                                     data_idx=cv_val_idx,
    #                                     shuffle_phenos=False,
    #                                     include_lineage=include_lineage,
    #                                     include_peptide_lengths=include_peptide_lengths,
    #                                     include_protein_embeddings=include_protein_embeddings,
    #                                     bounded_loss=bounded_loss,
    #                                     batch_size=BATCH_SIZE,
    #                                     shuffle_batches=False, # don't need to shuffle validation data because order doesn't matter,
    #     )
    
    #     # all MLP features are combined into the same vector, so if at least one of them is True, there is a vector
    #     # first index: batches, so take any one. Here, I took index 0
    #     # second index: (CNN_inputs, MIC_outputs), so take index 0
    #     # third index: inputs, can be of length 3-4. If length 3, there is nucleotide matrix, lower bounds, and upper bounds. If length 4, there is also MLP input at index 1
    #     # fourth index: samples in a single batch, so take any one. Here, I took index -1
    #     if include_lineage or include_peptide_lengths or include_protein_embeddings:
    #         additional_data_len = cv_val_generator[0][0][1].shape[1]
    #     else:
    #         additional_data_len = 0
    
    #     if drug == "PZA":
    #         patience_epochs = 75
    #     else:
    #         patience_epochs = 50
        
    #     for alpha in reg_param_lst:
    
    #         print(f"\nTraining split {split} on {len(cv_train_idx)} isolates and validating on {len(cv_val_idx)} isolates with alpha = {alpha}")
            
    #         # initialize model
    #         model = conv_nn(binary, longest_locus, num_loci, additional_data_len, bounded_loss, filter_size, reg_strength=alpha)
        
    #         # need to set save_history_df=True so that the history dataframe is saved, instead of returning the full array of losses
    #         # then you can return just the final loss using return_min_loss=True
    #         model_loss = train_single_CNN(model, loss_type, N_epochs, cv_train_generator, cv_val_generator, len(cv_train_idx), len(cv_val_idx), save_model_fName=os.path.join(output_path, "best_model.h5"), save_history_df=True, patience_epochs=patience_epochs, return_min_loss=True)
    #         print(model_loss)
            
    #         K.clear_session()
    
    #         losses_df = pd.concat([losses_df, pd.DataFrame({"alpha": alpha, "split": split, "val_loss": model_loss}, index=[0])])

    # print(f"Finished performing cross-validation for {output_path}")
    # losses_df.to_csv(os.path.join(output_path, "reg_param_losses.csv"), index=False)
    # losses_df = pd.read_csv(os.path.join(output_path, "reg_param_losses.csv"))
    
    # # get average loss across the 5 splits for a given regularization parameter, then get the param with the smallest average loss across the split
    # losses_df_grouped_alpha = pd.DataFrame(losses_df.groupby("alpha")["val_loss"].mean()).reset_index().rename(columns={"index": "alpha"})
    # select_alpha = losses_df_grouped_alpha.sort_values("val_loss", ascending=True)["alpha"].values[0]    
    # print(f"    Regularization parameter: {select_alpha}, minimum average validation loss across CV splits: {losses_df_grouped_alpha.sort_values('val_loss', ascending=True)['val_loss'].values[0]}")
    