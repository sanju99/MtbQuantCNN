# create PCA-transformed embeddings file
if include_protein_embeddings and not os.path.isfile(protein_embeddings_PC_fName):

    # first locate the associated PCA model. It's one level up (not in the TRUST or inSilico_analysis directories)
    pca = pickle.load(open(f"{os.path.dirname(seq_data_path)}/PCA_{N_PC}.sav", "rb"))
    
    # combine the embeddings for all proteins into a single file. Read them in in the same order as embed_genes_list for consistency with training data. THIS IS REQUIRED
    df_embeddings = pd.concat([pd.read_csv(f"{data_dir}/embeddings/{gene}.csv.gz", compression='gzip', index_col=[0]) for gene in embed_genes_list], axis=1, ignore_index=False)

    # Embeddings file includes H37Rv, so need to remove it by putting it in the same order as df_samples
    df_embeddings = df_embeddings.loc[df_samples.ROLLINGDB_ID.values]

    # transform the validation data by multiplying the original data (standard-scaled first, just like we did before fitting PCA) by the eigenvectors (transpose to match dimensions)
    X_pca = np.dot(scaler.fit_transform(df_embeddings.values), np.transpose(pca.components_))

    # save
    X_pca = pd.DataFrame(X_pca)
    X_pca.columns = [f"PC{num}" for num in X_pca.columns]
    X_pca['ROLLINGDB_ID'] = df_embeddings.index.values
    
    # add lineages and MICs to the dataframe. Merge left to keep H37Rv in the dataframe. Need the embeddings later for doing in silico validation and saliency score computation
    X_pca = X_pca.merge(df_samples, on='ROLLINGDB_ID', how='left')
    
    # keep index because the sample IDs are there
    X_pca.set_index('ROLLINGDB_ID').to_csv(protein_embeddings_PC_fName)