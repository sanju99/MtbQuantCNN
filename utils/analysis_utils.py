import numpy as np
import pandas as pd
import os, glob, sparse, yaml
from Bio import SeqIO
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import sklearn.utils
import scipy.stats as st

from data_utils import *


drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETH",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMI",
                  "Pretomanid": "PTM",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LEV",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

abbr_drug_dict = {value: key for key, value in drug_abbr_dict.items()}




def mutation_catalog_with_bootstrapping(df, drug, drug_abbr, who_variants_df, isolate_variants_df, binary_thresh, return_stats, savefName, AF_thresh=0.75):

    # make AF column float so that you can use AF thresh
    isolate_variants_df['AF'] = isolate_variants_df['AF'].replace('.', 0.76).astype(float)
    
    df = df.rename(columns={"ROLLINGDB_ID": "Isolate"}).reset_index(drop=True)
    highConf_mutations = who_variants_df.query("drug == @drug_abbr & confidence in ['1) Assoc w R', '2) Assoc w R - Interim']").mutation.values
    isolates_R = isolate_variants_df.query("mutation in @highConf_mutations & FILTER in ['PASS', 'Amb'] & AF >= @AF_thresh & Isolate in @df.Isolate.values").Isolate.values
        
    df_pred_catalog = df[["Isolate", f"{drug_abbr}_upper_bound"]]
    df_pred_catalog["y_test"] = (df[f"{drug_abbr}_upper_bound"] > binary_thresh).astype(int)
    df_pred_catalog["y_pred"] = df_pred_catalog["Isolate"].map(dict(zip(isolates_R, np.ones(len(isolates_R))))).fillna(0).astype(int)
    df_pred_catalog.to_csv(savefName, index=False)
    
    df_stats = compute_binary_metrics(df_pred_catalog["y_test"], df_pred_catalog["y_pred"], binary_thresh, binarize=False)[return_stats]
    df_stats["CV"] = 0
    bs_lst = []
    
    # perform bootstrapping with 10 replicates
    for i in range(10):
        
        bs_sample_idx = np.random.choice(df.index.values, size=len(df), replace=True)
        bs_df = df.iloc[bs_sample_idx, :]
        bs_isolates_R = isolate_variants_df.query("mutation in @highConf_mutations & FILTER in ['PASS', 'Amb'] & AF >= @AF_thresh & Isolate in @bs_df.Isolate.values").Isolate.values
        
        bs_pred_catalog = bs_df[["Isolate", f"{drug_abbr}_upper_bound"]]
        bs_pred_catalog["y_test"] = (bs_df[f"{drug_abbr}_upper_bound"] > binary_thresh).astype(int)
        bs_pred_catalog["y_pred"] = bs_pred_catalog["Isolate"].map(dict(zip(bs_isolates_R, np.ones(len(bs_isolates_R))))).fillna(0).astype(int)
        
        bs_df_stats = compute_binary_metrics(bs_pred_catalog["y_test"], bs_pred_catalog["y_pred"], binary_thresh, binarize=False)[return_stats]
        bs_df_stats["CV"] = i + 1
        bs_lst.append(bs_df_stats)

    df_return = pd.concat([df_stats, pd.concat(bs_lst, axis=0)], axis=0).reset_index(drop=True)
    df_return["Model"] = "Catalog"
    return df_return



def classify_using_mutation_catalog(config_file, who_variants_df, valOnlykeepidx=None, return_stats=["Sensitivity", "Specificity", "Precision", "Accuracy", "Balanced_Acc"], AF_thresh=0.75):

    kwargs = yaml.safe_load(open(config_file, "r"))

    df_phenos = pd.read_csv(kwargs["phenotype_file"])
    data_path = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    drug_abbr = kwargs["drug"]
    drug = abbr_drug_dict[drug_abbr]
    output_path = kwargs["output_path"]
    binary_thresh = kwargs["binary_thresh"]
    include_lineage = kwargs["include_lineage"]
    isolate_variants_df = pd.read_csv(os.path.join(data_path, "isolate_variants_fixed_annot.csv"))

    if AF_thresh == 0.25:
        suffix = "_lowAF"
    else:
        suffix = ""
    
    df_train = df_phenos.query("category=='original_train_set'")
    df_test = df_phenos.query("category=='original_test_set'")

    if os.path.isfile(os.path.join(data_path, "validation_data_for_model.csv")):
        include_val = True
        df_val = pd.read_csv(os.path.join(data_path, "validation_data_for_model.csv"))
    else:
        include_val = False

    if valOnlykeepidx is None:
        df_train = mutation_catalog_with_bootstrapping(df_train, drug, drug_abbr, who_variants_df, isolate_variants_df, binary_thresh, return_stats, os.path.join(output_path, f"catalog_train_predictions{suffix}.csv"), AF_thresh=AF_thresh)
        df_train["Dataset"] = "Train"
        
        df_test = mutation_catalog_with_bootstrapping(df_test, drug, drug_abbr, who_variants_df, isolate_variants_df, binary_thresh, return_stats, os.path.join(output_path, f"catalog_test_predictions{suffix}.csv"), AF_thresh=AF_thresh)
        df_test["Dataset"] = "Test"

        if include_val:
            df_val = mutation_catalog_with_bootstrapping(df_val, drug, drug_abbr, who_variants_df, isolate_variants_df, binary_thresh, return_stats, os.path.join(output_path, f"catalog_validation_predictions{suffix}.csv"), AF_thresh=AF_thresh)
            df_val["Dataset"] = "Validation"
            
            return pd.concat([df_train, df_test, df_val], axis=0).reset_index(drop=True)
        else:
            return pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
            
    else:
        # only return results for the validation set
        df_val = mutation_catalog_with_bootstrapping(df_val.iloc[valOnlykeepidx], drug, drug_abbr, who_variants_df, isolate_variants_df, binary_thresh, return_stats, os.path.join(output_path, f"catalog_validation_predictions{suffix}.csv"), AF_thresh=AF_thresh)
        df_val["Dataset"] = "Validation"
        return df_val

    

def create_summary_df(df_test, y_pred, drug, binary_thresh, num_loci, model_name, binarize=True, save_fName=None):
    
    # predictions dataframe: get indices of validation data in the cv splits
    pred_df = df_test[["ROLLINGDB_ID", f"{drug}_midpoint", f"{drug}_lower_bound", f"{drug}_upper_bound"]]

    # rename columns to make them easier to read
    pred_df.rename(columns={"ROLLINGDB_ID": "Isolate", 
                            f"{drug}_midpoint": "y_test",
                            f"{drug}_lower_bound": "lower",
                            f"{drug}_upper_bound": "upper"
                           }, 
                   inplace=True
                  )

    # add model predictions, and log-transform the test values
    pred_df["y_pred"] = np.squeeze(y_pred)
    pred_df["y_test"] = np.log2(pred_df["y_test"])

    binned_mae, binned_mse = boundedLoss_predict(pred_df, binary_thresh)
    
    if save_fName is not None:
        pred_df.to_csv(save_fName, index=False)

    summary_df = pd.DataFrame({"Drug": drug,
                               "Model": model_name,
                               "Num_Loci": num_loci,
                               "Binned_MAE": binned_mae,
                               "Binned_MSE": binned_mse,
                               "MAE": np.mean(np.abs(pred_df["y_pred"]-pred_df["y_test"])),
                               "MSE": np.mean(np.square(pred_df["y_pred"]-pred_df["y_test"])),
                               # "Within_doubling": within_doubling,
                               "Spearman": st.spearmanr(pred_df["y_pred"], pred_df["y_test"])[0],
                               "Pearson": st.pearsonr(pred_df["y_pred"], pred_df["y_test"])[0],
                              }, index=[0])

    # compute binary metrics using the upper bound
    binary_metrics_df = compute_binary_metrics(pred_df["upper"], pred_df["y_pred"], binary_thresh, binarize=binarize)
    summary_df = pd.concat([summary_df, binary_metrics_df], axis=1)
    return summary_df





def plot_histories(path, binary=False, patience_epochs=25, replicates=True, saveName=None):
        
    if binary:
        prefix = "binary_"
    else:
        prefix = ""

    if replicates:
        histories = pd.read_csv(os.path.join(path, f"bootstrapping/{prefix}histories.csv"))
    
    history = pd.read_csv(os.path.join(path, f"{prefix}history.csv"))
    history = history.iloc[:-patience_epochs, :]
    
    fig, ax = plt.subplots(figsize=(10, 4))

    if replicates:
        for col in histories.columns:
            single_rep = histories[[col]]
            single_rep = single_rep.loc[~pd.isnull(single_rep[col])].reset_index(drop=True)
            single_rep = single_rep.iloc[:-patience_epochs, :]
            print(f"Trained replicate {col} for {len(single_rep)} epochs (excluding patience period)")
            
            plt.plot(single_rep.index.values + 1, single_rep[col], color="lightgray")

    # plt.plot(histories.index + 1, histories.mean(axis=1), color="red", linewidth=1.5)
    plt.plot(history.index + 1, history["val_loss"], color="red", linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plot_title = "Tuning Loss" + " for " + os.path.basename(path) + " Model"
    plt.title(plot_title, fontsize=14)
    sns.despine()
    
    if saveName is None:
        plt.show()
    else:
        if not os.path.isdir(os.path.dirname(saveName)):
            os.makedirs(os.path.dirname(saveName))
        plt.savefig(saveName, dpi=300)


        


def binary_model_analysis(path, drug, cc, lineage=0, cv=True, plot=True, patience=25):
    
    history = pd.read_csv(os.path.join(path, "binary_history.csv"))
    pred = pd.read_csv(os.path.join(path, "binary_test_predictions.csv"))
    
    if patience != 0:
        history = history.iloc[:-patience, :]

    print(f"{len(pred)} points in validation set")
    print(f"Trained CNN for {len(history)} epochs")
    
    if plot:
        fig, ax = plt.subplots(figsize=(4, 2))

        plt.plot(history.index.values+1, history["val_loss"].values, label="Loss")
        plt.title("Validation Loss", fontsize=14)
        #plt.xlim(0, stop_epoch)

        sns.despine()
        plt.xlabel("Epoch", fontsize=10)
    
    summary_df = pd.DataFrame({"Drug": drug,
                               "Model": "CNN",
                              }, index=[0])
    
    summary_df["Num_Loci"] = "Binary"
    summary_df["CV"] = 0
    summary_df["Lineage"] = lineage
    summary_metrics_df = compute_binary_metrics(pred["y_pred_label"].values, pred["y_test"].values, cc, binarize=False)
    summary_df = pd.concat([summary_df, summary_metrics_df], axis=1)
    
    if cv:
        cv_results = pd.read_csv(os.path.join(path, "binary_val_results.csv"))
        cv_results["CV"] = np.arange(len(cv_results))+1
        cv_results["Num_Loci"] = "Binary"
        cv_results["Lineage"] = lineage
        return pd.concat([summary_df, cv_results])
    else:
        return summary_df
    

def quant_model_analysis(output_path, drug, include_lineage=False, include_peptide_length=False, bs=True, plot=True, patience=25, suffix="", saveName=None):

    print(output_path)
    pred_df = pd.read_csv(os.path.join(output_path, f"test_predictions{suffix}.csv"))
    history = pd.read_csv(os.path.join(output_path, "history.csv"))
    
    if patience != 0:
        last_epoch = len(history) - patience
    else:
        last_epoch = len(history)
    
    # print(f"{len(pred_df)} points in validation set")
    # print(f"Trained CNN for {last_epoch} epochs")
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

        ax[0].plot(history.index.values+1, history["val_loss"].values, label="Loss")
        ax[0].set_title("Validation Loss", fontsize=14)
        ax[0].set_xlabel("Epoch", fontsize=10)
        ax[0].axvline(x=last_epoch, color="black", linestyle="dashed")
        
        values = np.vstack([pred_df["y_pred"], pred_df["y_test"]])
        kernel = st.gaussian_kde(values)(values)

        sns.scatterplot(
            data=pred_df,
            x="y_pred",
            y="y_test",
            c=kernel,
            cmap="Blues",
            linewidth=0.25,
            edgecolor="lightgray",
            ax=ax[1]
        )

        max_val = np.max([pred_df.y_pred, pred_df.y_test]) * 1.1
        min_val = np.min([pred_df.y_pred, pred_df.y_test]) * 1.1

#         ax[1].set_xscale(matplotlib.scale.LogScale(ax[1], base=2))
#         ax[1].set_yscale(matplotlib.scale.LogScale(ax[1], base=2))
        
        ax[1].set_xlabel("Predicted MIC (µg/mL)")
        ax[1].set_ylabel("Actual MIC (µg/mL)")
        ax[1].set_xlim(min_val, max_val)
        ax[1].set_ylim(min_val, max_val)
        
        tick_labels = []

        for num in np.exp2(ax[1].get_xticks()):
            if num < 1:
                tick_labels.append(str(np.round(num, 2)))
            else:
                tick_labels.append(str(int(num)))
        
        ax[1].set_xticks(ticks=ax[1].get_xticks(), labels=tick_labels)
        ax[1].set_yticks(ticks=ax[1].get_yticks(), labels=tick_labels)
        
        ax[1].set_xlabel("\nPredicted MIC (µg/mL)", fontsize=9)
        ax[1].set_ylabel("Actual MIC (µg/mL)\n", fontsize=9)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    
        sns.despine()
        plt.tight_layout()
        
        if saveName is None:
            plt.show()
        else:
            if not os.path.isdir(os.path.dirname(saveName)):
                os.makedirs(os.path.dirname(saveName))
            plt.savefig(saveName, dpi=300)
            
    summary_df = pd.read_csv(os.path.join(output_path, f"results{suffix}.csv"))
    summary_df["Lineage"] = int(include_lineage)
    summary_df["Peptide"] = int(include_peptide_length)
    
    if bs:
        bs_results = pd.read_csv(os.path.join(output_path, f"bootstrapping/results{suffix}.csv"))
        bs_results["Lineage"] = int(include_lineage)
        bs_results["Peptide"] = int(include_peptide_length)
        summary_df = pd.concat([summary_df, bs_results])

    del_cols = ["AUC", "AUC_PR"]

    for col in del_cols:
        if col in summary_df.columns:
            del summary_df[col]

    return summary_df




def get_train_test_val_lineages(df_train, df_test, df_val=None, lineage_fName="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineage_matrix_Coll2014.csv"):
    
    lineages = pd.read_csv(lineage_fName, index_col=[0])
    lineages.columns = [f"lineageSNP_{col}" for col in lineages.columns]
    assert len(np.unique(lineages.values)) == 2

    train_lineages = lineages.loc[df_train["ROLLINGDB_ID"].values]
    assert sum(train_lineages.index.values != df_train["ROLLINGDB_ID"].values) == 0

    test_lineages = lineages.loc[df_test["ROLLINGDB_ID"].values]
    assert sum(test_lineages.index.values != df_test["ROLLINGDB_ID"].values) == 0

    # include support for no validation data (i.e. Pyrazinamide)
    if df_val is not None:
        val_lineages = lineages.loc[df_val["ROLLINGDB_ID"].values]
        assert sum(val_lineages.index.values != df_val["ROLLINGDB_ID"].values) == 0
    else:
        val_lineages = None
        
    return train_lineages, test_lineages, val_lineages
    



def prepare_model_inputs(X, model_type, include_lineage, feature_names=None, lineages_matrix=None):
    
    if model_type == "CNN":
        
        if include_lineage:
            model_inputs = [X, lineages_matrix.values, np.zeros(len(X)), np.zeros(len(X))]
        else:
            model_inputs = [X, np.zeros(len(X)), np.zeros(len(X))]
        
    elif model_type == "Regression":
  
        if include_lineage:
            
            # # first combine the dataframes to preserve the features, then get only the features specified in the argument
            # assert sum(X.index.values != lineages_matrix.index.values) == 0
            model_inputs = X.merge(lineages_matrix, left_index=True, right_index=True, how="inner")
            model_inputs = model_inputs[feature_names].values
        else:
            # keep only the features specified in the argument
            model_inputs = X[feature_names].values
        
    else:
        raise ValueError(f"{model_type} is not a valid model type!")
        
    return model_inputs

    


def get_inputs_for_regression(config_file):

    kwargs = yaml.safe_load(open(config_file, "r"))

    df_phenos = pd.read_csv(kwargs["phenotype_file"])
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    locus_list = kwargs["locus_list"]
    drug = kwargs["drug"]
    results_dir = kwargs["output_path"]
    ridge_dir = os.path.join(results_dir, "ridge")
    fasta_dir = kwargs["genotype_input_directory"]
    include_lineage = kwargs["include_lineage"]

    # make dataframes of coordinates
    gene_coords, _ = get_gene_coords(locus_list, fasta_dir)
    h37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)    

    # this is for samples that don't have data from the MIC-ML consortium, so there is no validation dataset
    if os.path.isfile(os.path.join(data_dir, "validation_data_for_model.csv")):
        val_data_present = True
    else:
        val_data_present = False

    if val_data_present:

        df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))

        # get the pickle file made for the CNN input
        if not os.path.isfile(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")):
            
            val_matrix = get_new_aln_for_CNN(df_val,
                                            locus_list,
                                            fasta_dir
                                           )
            sparse.save_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz"), sparse.COO(val_matrix))
        else:
            val_matrix = sparse.load_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")).todense()
        
        val_samples = val_matrix.shape[0]  
        one_hot_encodings = val_matrix.shape[1]
        longest_locus = val_matrix.shape[2]
        num_loci = val_matrix.shape[3]
        assert one_hot_encodings == 5

        ref_matrix = sparse.load_npz(f"{results_dir.replace('_lineage', '')}/pkl_sparse_ref.npz").todense()
        
        if os.path.isfile(os.path.join(ridge_dir.replace("_lineage", ""), "val_seq_matrix.pkl")):
            X_val = pd.read_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "val_seq_matrix.pkl"))
    
        else:
            print(f"Creating validation data pickle file")
            
            X_val = []
            
            for locus in locus_list:

                # don't need the reference matrix here
                single_locus_matrix, _ = get_single_locus_Reg_input(locus, locus_list, df_phenos, val_matrix, ref_matrix, h37Rv_coords)
                X_val.append(single_locus_matrix)
        
            X_val = pd.concat(X_val, axis=1)
            X_val.index = df_val["ROLLINGDB_ID"].values
            X_val.to_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "val_seq_matrix.pkl"))

    else:
        df_val = None
        X_val = None

    # read in the pickle file of all the sequence features. This should be of length 5 x ALL nucleotides across all loci
    # this is before anything has been dropped due to redundancy or not being present in the samples
    X_train_test = pd.read_pickle(os.path.join(ridge_dir.replace("_lineage", ""), "full_seq_matrix.pkl"))

    df_train = df_phenos.query("category=='original_train_set'").reset_index(drop=True)    
    df_test = df_phenos.query("category=='original_test_set'").reset_index(drop=True)    

    X_train = X_train_test.loc[df_train.ROLLINGDB_ID.values]
    X_test = X_train_test.loc[df_test.ROLLINGDB_ID.values]

    # X_train, X_test, and X_val should all be dataframes read in from pickle files, so the indices are ROLLLINGDB_ID and the columns are features
    return X_train, X_test, X_val, df_train, df_test, df_val



def get_new_aln_for_CNN(df,
                        locus_list,
                        fasta_dir
                       ):
    
    # argument = directory that contains the fasta file
    df_genos = make_genotype_df(locus_list, fasta_dir)
    df_genos.index = [name.split(".")[0] for name in df_genos.index.values]
        
    # the additional new strains to predict MICs for
    df_genos = df_genos.loc[df["ROLLINGDB_ID"].values]
    
    assert len(df_genos) == len(df)

    # Apply one-hot encoding function to get each isolate sequence
    print('making one hot encoding for...')
    for locus in locus_list:
        print("...", locus)
        lengths = [len(seq) for seq in df_genos[locus]]
        assert len(np.unique(lengths)) == 1
        df_genos[locus + "_one_hot"] = df_genos[locus].apply(np.vectorize(get_one_hot))
        
    return create_X(df_genos)


                           

def get_inputs_for_CNN(config_file, keep_idx=None):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    
    data_dir = os.path.dirname(kwargs["phenotype_file"])
    drug = kwargs["drug"]
    locus_list = kwargs["locus_list"]
    results_dir = kwargs["output_path"]
    fasta_dir = kwargs["genotype_input_directory"]
    include_lineage = kwargs["include_lineage"]
    df_phenos = pd.read_csv(kwargs["phenotype_file"])

    binary_thresh = kwargs["binary_thresh"]
    binary = kwargs["binary"]

    # this is for samples that don't have data from the MIC-ML consortium, so there is no validation dataset
    if os.path.isfile(os.path.join(data_dir, "validation_data_for_model.csv")):
        val_data_present = True

        df_val = pd.read_csv(os.path.join(data_dir, "validation_data_for_model.csv"))

        if not os.path.isfile(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")):
            
            X_val = get_new_aln_for_CNN(df_val,
                                        locus_list,
                                        fasta_dir
                                       )
            sparse.save_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz"), sparse.COO(X_val))
            
        else:
            X_val = sparse.load_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_val.npz")).todense()

    else:
        val_data_present = False
        df_val = None
        X_val = None
        
    df_train = df_phenos.query("category=='original_train_set'")
    df_test = df_phenos.query("category=='original_test_set'")    

    X_train_test = sparse.load_npz(os.path.join(results_dir.replace("_lineage", ""), "pkl_sparse_full.npz")).todense()
    X_train = X_train_test[df_train.index.values]
    X_test = X_train_test[df_test.index.values]

    # these are in the same order as df_train, df_test, and df_val, which are in the same order as X_train, X_test, and X_val
    train_lineages, test_lineages, val_lineages = get_train_test_val_lineages(df_train, df_test, df_val)

    X_train = prepare_model_inputs(X_train, "CNN", include_lineage, feature_names=None, lineages_matrix=train_lineages)
    X_test = prepare_model_inputs(X_test, "CNN", include_lineage, feature_names=None, lineages_matrix=test_lineages)
    
    if val_data_present:
        if keep_idx is not None:
            X_val = X_val[keep_idx, :]

            # lineages matrices have samples as the index for merging
            if include_lineage:
                val_lineages = val_lineages.iloc[keep_idx, :]

            df_val = df_val.iloc[keep_idx, :]
            
        X_val = prepare_model_inputs(X_val, "CNN", include_lineage, feature_names=None, lineages_matrix=val_lineages)

    # X_train, X_test, and X_val should all be numpy arrays (so no indices or columns)
    return X_train, X_test, X_val, df_train.reset_index(drop=True), df_test.reset_index(drop=True), df_val

# def get_all_higher_lineages(lineage):

#     if "," in lineage:
#         raise ValueError("Lineage entry can't have multiple lineages")
                
#     if "." in lineage:

#         hierarchical_lineages = []
#         split_lineages = lineage.split(".")

#         for k in range(len(split_lineages)):

#             hierarchical_lineages.append(".".join(split_lineages[:k+1]))

#         # check that there are N + 1 lineages in the list, where N is the number of divisions (.)
#         assert len(hierarchical_lineages) == lineage.count(".") + 1
#         return hierarchical_lineages

#     # nothing to split, so return a list with the input lineage
#     else:
#         return [lineage]

    

# def get_lineages_matrix(df_phenos):
    
#     lineage_indicator_df = pd.get_dummies(df_phenos[["Lineage"]], prefix="", prefix_sep="")
    
#     # check that before filling in the hierarchical lineages, each sample has a single lineage
#     assert len(np.unique(lineage_indicator_df.sum(axis=1))) == 1
#     assert np.unique(lineage_indicator_df.sum(axis=1))[0] == 1

#     for i, col in enumerate(lineage_indicator_df.columns):

#         hierarchical_lineages = get_all_higher_lineages(col)

#         found_cols = []
#         primary_lineage = hierarchical_lineages[0]

#         # the primary lineage should definitely be in the dataframe
#         assert primary_lineage in lineage_indicator_df.columns

#         for col in hierarchical_lineages:
#             if col in lineage_indicator_df.columns:
#                 found_cols.append(col)

#         # remove the last lineage from the list (this column already has 1)
#         final_lineage = found_cols.pop(-1)

#         lineage_indicator_df.loc[lineage_indicator_df[final_lineage]==1, found_cols] += 1

#     # check that every row (sample) has at least one lineage and every column (lineage) has at least one isolate
#     assert np.min(lineage_indicator_df.sum(axis=1)) >= 1
#     assert np.min(lineage_indicator_df.sum(axis=0)) >= 1

#     lineage_indicator_df.index = df_phenos["ROLLINGDB_ID"].values

#     return lineage_indicator_df