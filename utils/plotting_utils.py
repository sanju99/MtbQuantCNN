import numpy as np
import pandas as pd
import os, glob, sparse, yaml
from Bio import SeqIO
BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}
import sklearn.metrics
import sklearn.utils
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns



def plot_histories(path, patience_epochs=200, cv=True, saveName=None):

    if cv:
        combined_replicates = [pd.read_csv(fName)[['val_loss']] for fName in glob.glob(os.path.join(path, f"cross_validation/history*.csv"))]
    
    history = pd.read_csv(os.path.join(path, "history.csv"))
    history = history.iloc[:-patience_epochs, :]
    
    fig, ax = plt.subplots(figsize=(10, 4))

    if cv:
        for i, single_rep in enumerate(combined_replicates):
            single_rep = single_rep.iloc[:-patience_epochs, :]
            print(f"Trained replicate {i} for {len(single_rep)} epochs")
            
            plt.plot(single_rep.index.values + 1, single_rep['val_loss'], color="lightgray")

    plt.plot(history.index + 1, history["val_loss"], color="red", linewidth=1.5)
    print(f"Trained full model for {len(history)} epochs")

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