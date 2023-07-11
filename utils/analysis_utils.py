import numpy as np
import pandas as pd
import os, glob, sparse
from Bio import SeqIO
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import sklearn.utils
import scipy.stats as st

from data_utils import *



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

    binned_mae, binned_mse, within_1bin, within_doubling = boundedLoss_predict(pred_df, binary_thresh)
    
    if save_fName is not None:
        pred_df.to_csv(save_fName, index=False)

    summary_df = pd.DataFrame({"Drug": drug,
                               "Model": model_name,
                               "Num_Loci": num_loci,
                               "Binned_MAE": binned_mae,
                               "Binned_MSE": binned_mse,
                               "MAE": np.mean(np.abs(pred_df["y_pred"]-pred_df["y_test"])),
                               "MSE": np.mean(np.square(pred_df["y_pred"]-pred_df["y_test"])),
                               "Within_1Bin": within_1bin,
                               "Within_doubling": within_doubling,
                               "Spearman": st.spearmanr(pred_df["y_pred"], pred_df["y_test"])[0],
                               "Pearson": st.pearsonr(pred_df["y_pred"], pred_df["y_test"])[0],
                              }, index=[0])

    binary_metrics_df = compute_binary_metrics(pred_df["y_test"], pred_df["y_pred"], binary_thresh, binarize=binarize)
    summary_df = pd.concat([summary_df, binary_metrics_df], axis=1)
    return summary_df





def plot_histories(path, rep_CV=False, binary=False, patience_epochs=25, saveName=None):
        
    if binary:
        prefix = "binary_"
    else:
        prefix = ""
        
    if rep_CV:
        middle_part = "cv"
    else:
        middle_part = "bs"
        
    histories = pd.read_csv(os.path.join(path, f"bootstrapping/{prefix}history_{middle_part}_replicates.csv"))
    history = pd.read_csv(os.path.join(path, f"{prefix}history.csv"))
    history = history.iloc[:-patience_epochs, :]
    
    fig, ax = plt.subplots(figsize=(10, 4))

    for col in histories.columns:
        single_rep = histories[[col]]
        single_rep = single_rep.loc[~pd.isnull(single_rep[col])].reset_index(drop=True)
        #single_rep = single_rep.iloc[:-patience_epochs, :]
        
        plt.plot(single_rep.index.values + 1, single_rep[col], color="lightgray")

    # plt.plot(histories.index + 1, histories.mean(axis=1), color="red", linewidth=1.5)
    plt.plot(history.index + 1, history["val_loss"], color="red", linewidth=1.5)

    plt.xlabel("Epoch", fontsize=12)
    plot_title = "Validation Loss" + " for " + os.path.basename(path) + " Model"
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
    

def quant_model_analysis(path, drug, lineage=0, cv=True, plot=True, patience=25, saveName=None):
    
    pred_df = pd.read_csv(os.path.join(path, "test_predictions.csv"))
    history = pd.read_csv(os.path.join(path, "history.csv"))
    
    if patience != 0:
        last_epoch = len(history) - patience
    else:
        last_epoch = len(history)
    
    print(f"{len(pred_df)} points in validation set")
    print(f"Trained CNN for {last_epoch} epochs")
    
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
            
    summary_df = pd.read_csv(os.path.join(path, "cnn_results.csv"))
    summary_df["Lineage"] = lineage
    
    if cv:
        cv_results = pd.read_csv(os.path.join(path, "bootstrapping/bs_results.csv"))
        cv_results["Lineage"] = lineage
        summary_df = pd.concat([summary_df, cv_results])

    del_cols = ["AUC", "AUC_PR"]

    for col in del_cols:
        if col in summary_df.columns:
            del summary_df[col]

    return summary_df


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