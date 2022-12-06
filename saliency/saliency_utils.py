from dna_features_viewer import BiopythonTranslator, GraphicFeature, GraphicRecord
from dna_features_viewer.biotools import annotate_biopython_record
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools
from Bio import SeqIO
from Bio.Seq import Seq
from cnn_utils import MtbGeneDataset
import scipy.stats as st


# import reference files
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")

freschi_snps = pd.read_excel("../lasso/Freschi_SNPs.xlsx")
freschi_snps["position"] = freschi_snps["position"].astype(int)


# create plotting class
class MyCustomTranslator(BiopythonTranslator):
    """Custom translator implementing the following theme:

    - Color genes in blue
    , promoters in pink
    """

    def compute_feature_color(self, feature):
        if feature.type == "CDS":
            return "blue"
        elif feature.type == "promoter":
            return "#fbb4ae"
        else:
            return "#b3cde3"

    def compute_feature_label(self, feature):
        if feature.type == "CDS":
            return "CDS here"
        elif feature.type == "promoter":
            # text returned here is the label displayed in the figure
            return None
        else:
            return BiopythonTranslator.compute_feature_label(self, feature)

    def compute_filtered_features(self, features):
        """Only display genes """
        return [
            feature for feature in features
            if (feature.type == "gene" or feature.type=="promoter")
        ]
    
#  H37Rv full genome genbank file
record = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
print(len(record.seq))

graphic_record = MyCustomTranslator().translate_record(record)

############ CUSTOM FUNCTIONS TO GENERATE SALIENCY PLOTS ##################

def get_gene_coords(locus_list, fasta_dir):
    '''
    Use this function to get a dataframe of coordinates from the bash scripts used to generate the alignment FASTA files for every locus.
    '''
    coords = []
    sense_lst = []
    
    for locus in locus_list:
        # read the coordinates from the file
        with open(os.path.join(fasta_dir, locus + ".sh"), "r") as file:
            for line in file:
                if line[0] not in ["#", "\n"]:
                    split_line = line.split("-")
                    coords.append([int(split_line[0].split(" ")[-1]), 
                                   int(split_line[1].split(" ")[0])
                                  ])
                    
                    # sense comes after the coordinates
                    sense_lst.append(split_line[1].split(" ")[1])
                    
    gene_coords = pd.DataFrame(coords)
    gene_coords.columns = ["Start", "End"]
    gene_coords["Locus"] = locus_list    
                    
    gene_coords["Length"] = gene_coords["End"] - gene_coords["Start"]
    gene_coords["Sense"] = sense_lst
    gene_coords = gene_coords.set_index("Locus")

    # during this iteration, convert everything to 1-indexing because using np.arange on inverted coordinates is going to get messy
    # so add 1 to the start position, and then both coordinates should be inclusive
    for i, row in gene_coords.iterrows():
        if row["Sense"] == "neg":
            new_start = row["End"]
            new_end = row["Start"] + 1
            gene_coords.loc[i, "Start"] = new_start
            gene_coords.loc[i, "End"] = new_end
        else:
            gene_coords.loc[i, "Start"] = row["Start"] + 1
            gene_coords.loc[i, "End"] = row["End"]
            
    assert sum(gene_coords.query("Sense=='neg'").End > gene_coords.query("Sense=='neg'").Start) == 0
    assert sum(gene_coords.query("Sense=='pos'").End < gene_coords.query("Sense=='pos'").Start) == 0

    return gene_coords, dict(zip(locus_list, sense_lst))



def make_h37rv_coordinates(gene_coords, locus_list, fasta_dir):
    '''
    gene_coords is 1-indexed, and for negative sense genes, start position is downstream of end position.
    '''
    dfs_list = []
    
    for locus in locus_list:
                
        # read in the sequences for the fasta file
        seqs = [(seq.id, seq.seq) for seq in SeqIO.parse(os.path.join(fasta_dir, f"{locus}.fasta"), "fasta")]
                
        # H37Rv is the last one
        H37Rv = list(seqs[-1][1])
        H37Rv_coords = pd.DataFrame(H37Rv).rename(columns={0:locus})

        # replace deletion characters with nan
        coords_count = []
        pos = gene_coords.loc[locus, "Start"]
        sense = gene_coords.loc[locus, "Sense"]
        length = gene_coords.loc[locus, "Length"]
        assert len(H37Rv) >= length

        for _, row in H37Rv_coords.iterrows():

            if row[locus] == "-":
                coords_count.append(np.nan)
            else:
                coords_count.append(pos)
                if sense == "pos":
                    pos += 1
                else:
                    pos -= 1
                    
        # check that the last number is the same as the end position
        assert pd.Series(coords_count).dropna()[:length].values[-1] == gene_coords.loc[locus, "End"]

        # combine the locus name with the coordinates and remove the sequence
        H37Rv_coords[locus + "_coord"] = coords_count
        del H37Rv_coords[locus]
        dfs_list.append(H37Rv_coords)
        
    # this is 1-indexed now and in reverse order for negative sense genes
    return pd.concat(dfs_list, axis=1).values



# def get_locus_adj_coords(X_matrix_H37Rv_coords, gene_coords, locus_list, locus, fasta_dir, combined_max, combined_min, combined_mean):
#     '''
#     Adjustments for insertions relative to H37Rv are made for all loci. Adjustments for deletions are only made for loci that don't have
#     deletions at the start and end. Deletions at the ends mean that deletions occurred outside the locus, so the saliencies have to remain
#     where they are because the coordinates are not extended past the locus. This may cause some difficulties for mapping positions back to
#     the original VCFs / isolate_variants file, so it'll just have to be searched manually for now. 
#     '''
#     idx = locus_list.index(locus)

#     seqs = [seq.seq for seq in SeqIO.parse(f"{fasta_dir}/{locus}.fasta", "fasta")]
#     H37Rv_locus = seqs[-1]
#     aln_len = len(H37Rv_locus)

#     # the alignment length should be the same or longer (indels) as the H37Rv reference sequence
#     assert aln_len >= gene_coords.loc[locus, "Length"]
    
#     new_coord_df = pd.DataFrame(columns=["Pos", "Type"])
    
#     # check that after the length of the alignment, the rest are padded positions (NaNs)
#     assert sum(~pd.isnull(X_matrix_H37Rv_coords[:, idx][aln_len:])) == 0

#     #print("Adjusting coordinates for insertions relative to H37Rv...")
#     for i, char in enumerate(list(H37Rv_locus)):
#         if char == "-":
#             # if the previous character is not a gap, then it's easy. Take that position
#             if H37Rv_locus[i-1] != "-":
#                 new_coord_df.loc[i] = [((X_matrix_H37Rv_coords[:, idx][:aln_len])[i-1]), "gap_adj"]
#             else:            
#                 # find the first occurrence of a gap between the previous new index and the current gap 
#                 new_coord_df.loc[i] = [new_coord_df.query("Type=='not_gap'").iloc[-1, :].Pos, "gap_far"]

#         # get the coordinate as usual
#         else:
#             new_coord_df.loc[i] = [((X_matrix_H37Rv_coords[:, idx][:aln_len])[i]), "not_gap"]
            
#     # make a copy to store means, then concatenate later
#     mean_coord_df = new_coord_df.copy()
    
#     # add max and min to the main dataframe
#     new_coord_df["Max"] = combined_max[:aln_len, idx]
#     new_coord_df["Min"] = combined_min[:aln_len, idx]

#     assert new_coord_df.Pos.values[0] == gene_coords.loc[locus, "Start"]
#     assert new_coord_df.Pos.values[-1] == gene_coords.loc[locus, "End"]

#     new_coord_df = new_coord_df.groupby("Pos")[["Max", "Min"]].sum().reset_index()
    
#     mean_coord_df["Mean"] = combined_mean[:aln_len, idx]
#     mean_coord_df = mean_coord_df.groupby("Pos")[["Mean"]].mean().reset_index()
        
#     new_coord_df = new_coord_df.merge(mean_coord_df, on="Pos")
#     new_coord_df["Pos"] = new_coord_df["Pos"].astype(int)
#     return new_coord_df




# def get_locus_adj_coords_with_deletions(X_matrix_H37Rv_coords, gene_coords, locus_list, locus, fasta_dir, combined_max, combined_min, combined_mean, isolate_variants_df):
#     '''
#     Adjustments for insertions relative to H37Rv are made for all loci. Adjustments for deletions are only made for loci that don't have
#     deletions at the start and end. Deletions at the ends mean that deletions occurred outside the locus, so the saliencies have to remain
#     where they are because the coordinates are not extended past the locus. This may cause some difficulties for mapping positions back to
#     the original VCFs / isolate_variants file, so it'll just have to be searched manually for now. 
#     '''
#     idx = locus_list.index(locus)
        
#     seqs = [(os.path.basename(seq.id).split(".")[0], str(seq.seq)) for seq in SeqIO.parse(f"{fasta_dir}/{locus}.fasta", "fasta")]
#     H37Rv_locus = seqs[-1][1]

#     aln_len = len(H37Rv_locus)
#     seq_df = pd.DataFrame(seqs).rename(columns={0:"Isolate", 1:"seq"})
#     isolates = seq_df.Isolate.values

#     seq_df = seq_df.seq.str.split("", expand=True).iloc[:, 1:-1].T.reset_index(drop=True)
#     seq_df.columns = isolates

#     # the alignment length should be the same or longer (indels) as the H37Rv reference sequence
#     assert aln_len >= gene_coords.loc[locus, "Length"]

#     del_idx = []
#     for i, row in seq_df.iterrows():

#         if ("-" in seq_df.loc[i, :].values) & (row["MT_H37Rv"] != "-"):
#             assert ~pd.isnull(i)
#             del_idx.append(int(i))

#     del_pos = X_matrix_H37Rv_coords[del_idx, idx]
    
#     # if there are huge deletions in any isolates, don't adjust for deletions below because it will throw off stuff
#     min_coord = np.min(gene_coords.loc[locus, ["Start", "End"]].values)
#     max_coord = np.max(gene_coords.loc[locus, ["Start", "End"]].values)
#     idx_large = np.where(isolate_variants_df.query("POS >= @min_coord & POS <= @max_coord")["REF_len"] - 
#                          isolate_variants_df.query("POS >= @min_coord & POS <= @max_coord")["ALT_len"] > 10)[0]
    
#     new_coord_df = pd.DataFrame(columns=["Pos", "Type"])
    
#     # check that after the length of the alignment, the rest are padded positions (NaNs)
#     assert sum(~pd.isnull(X_matrix_H37Rv_coords[:, idx][aln_len:])) == 0

#     #print("Adjusting coordinates for insertions relative to H37Rv...")
#     for i, char in enumerate(list(H37Rv_locus)):
#         curr_pos = (X_matrix_H37Rv_coords[:, idx][:aln_len])[i]
#         prev_pos = (X_matrix_H37Rv_coords[:, idx][:aln_len])[i-1]
        
#         if char == "-":
#             # if the previous character is not a gap, then it's easy. Take that position
#             if H37Rv_locus[i-1] != "-":
#                 new_coord_df.loc[i] = [prev_pos, "gap_adj"]
#             else:            
#                 # find the first occurrence of a gap between the previous new index and the current gap 
#                 new_coord_df.loc[i] = [new_coord_df.query("Type=='not_gap'").iloc[-1, :].Pos, "gap_far"]
        
#         else:
#             if (len(idx_large) == 0) & (curr_pos in del_pos) and (gene_coords.loc[locus, "Start"] not in del_pos) and (gene_coords.loc[locus, "End"] not in del_pos):
#                 if prev_pos not in del_pos:
#                     new_coord_df.loc[i] = [prev_pos, "gap_adj"]
#                 else:
#                     new_coord_df.loc[i] = [new_coord_df.query("Type=='gap_adj'").iloc[-1, :].Pos, "gap_far"]
#             # get the coordinate as usual
#             else:
#                 new_coord_df.loc[i] = [((X_matrix_H37Rv_coords[:, idx][:aln_len])[i]), "not_gap"]
                
#     # make a copy to store means, then concatenate later
#     mean_coord_df = new_coord_df.copy()
    
#     # add max and min to the main dataframe
#     new_coord_df["Max"] = combined_max[:aln_len, idx]
#     new_coord_df["Min"] = combined_min[:aln_len, idx]

#     assert new_coord_df.Pos.values[0] == gene_coords.loc[locus, "Start"]
#     assert new_coord_df.Pos.values[-1] == gene_coords.loc[locus, "End"]

#     new_coord_df = new_coord_df.groupby("Pos")[["Max", "Min"]].sum().reset_index()
    
#     mean_coord_df["Mean"] = combined_mean[:aln_len, idx]
#     mean_coord_df = mean_coord_df.groupby("Pos")[["Mean"]].mean().reset_index()
        
#     new_coord_df = new_coord_df.merge(mean_coord_df, on="Pos")
#     new_coord_df["Pos"] = new_coord_df["Pos"].astype(int)
    
#     # add additional positions that were dropped because they are deletions relative to H37Rv
#     # these are only positions that are not deletions in H37Rv
#     add_pos = set(np.arange(min_coord, max_coord+1)) - set(new_coord_df.Pos)

#     new_coord_df = pd.concat([new_coord_df, pd.DataFrame({"Pos": np.array(list(add_pos)).astype(int), "Max": np.zeros(len(add_pos)), "Min": np.zeros(len(add_pos))})])
#     assert len(new_coord_df) == gene_coords.loc[locus, "Length"]

#     return new_coord_df.sort_values("Pos")



def multi_locus_saliency(config_file, sense_dict, gene_coords, binary=False, isolate_variants_df=None, save=False):
    
    kwargs = yaml.safe_load(open(config_file, "r"))
    
    out_dir = kwargs["output_path"]
    fasta_dir = kwargs["genotype_input_directory"]
    drug = kwargs["drug"]
    locus_list = kwargs["locus_list"]
    df_phenos_path = kwargs["phenotype_file"]
    binary_thresh = kwargs["binary_thresh"]
    
    # this is 1-indexed and in reverse order for negative sense genes
    X_matrix_H37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)
    
    if binary:
        deeplift_dir = os.path.join(out_dir, "deeplift_outputs", "binary")
    else:
        deeplift_dir = os.path.join(out_dir, "deeplift_outputs", "quant")
    
    combined_mean = np.load(os.path.join(deeplift_dir, "deeplift_mean.npy"))
    combined_max = np.load(os.path.join(deeplift_dir, "deeplift_max.npy"))
    combined_min = np.load(os.path.join(deeplift_dir, "deeplift_min.npy"))
    
    # check signs
    assert np.max(combined_min.flatten()) <= 0
    assert np.min(combined_max.flatten()) >= 0
    
#     # get all scores for computing correlations
#     all_scores = sparse.load_npz(os.path.join(deeplift_dir, "scores_all_strains.npy.npz")).todense()
    
    # get MIC data for the traininset. Use the train generator for consistency in the order in which they were computed in run_deeplift.py
    # batch load data and compute saliency scores because inputs are too large, don't shuffle inputs
    train_generator = MtbGeneDataset(
        os.path.join(out_dir, 'pkl_sparse_train.npz'),
        df_phenos_path,
        kwargs["snp_table_file"],
        drug,
        locus_list,
        train_or_test="original_train_set",
        binary=binary,
        cc=binary_thresh,
        include_lineage=kwargs["include_lineage"],
        data_idx=None,
        batch_size=kwargs["batch_size"],
        shuffle=False
    )

    # get MICs for the training set
    mics = np.array([])

    for i, _ in enumerate(train_generator):
        batch = train_generator.__getTestData__(i)
        mics = np.concatenate([mics, batch[1]])
        
    #### DEFINE FUNCTION FOR COMPUTING CORRELATIONS BETWEEN MICS AND SALIENCIES OF THE TRAINING SET ####    
    
#     def compute_mic_saliency_correlation_single_locus(all_scores, locus, aln_len, locus_idx, locus_list, mic_lst):

#         single_locus_scores_all_strains = all_scores[:, :aln_len, locus_idx]

#         positive_corr = []

#         for idx in range(single_locus_scores_all_strains.shape[1]):
#             if len(np.unique(single_locus_scores_all_strains[:, idx])) > 1:
#                 corr = st.spearmanr(single_locus_scores_all_strains[:, idx], mic_lst)
#                 positive_corr.append([idx, corr[0], corr[1]])

#         df = pd.DataFrame(positive_corr)
#         df.columns = ["index", "Spearman_R", "pval"]
#         return df
            
    # initalize empty dataframe for saliency scores
    res_df = pd.DataFrame(columns=["Gene", "Pos", "Max", "Min"])

    fig, ax = plt.subplots(gene_coords.shape[0]*2, 1, figsize=(14, 3*gene_coords.shape[0]))
    axes=ax.flatten()

    #print("Computing new coordinates to account for indels...")
    for locus_idx, locus in enumerate(locus_list):

        #print(f"    {locus}")
        ax_coords = axes[(locus_idx)*2+1]
        ax_saliency = axes[(locus_idx)*2]

        # # this creates a dataframe of positions, Max, and Min saliency scores
        # if isolate_variants_df is not None:
        #     #print("Computing new coordinates to account for deletions relative to H37Rv....")
        #     new_coord_df = get_locus_adj_coords_with_deletions(X_matrix_H37Rv_coords, gene_coords, locus_list, locus, fasta_dir, combined_max, combined_min, combined_mean, isolate_variants_df)
        # else:
        #     new_coord_df = get_locus_adj_coords(X_matrix_H37Rv_coords, gene_coords, locus_list, locus, fasta_dir, combined_max, combined_min, combined_mean)
        
        # start = new_coord_df.Pos.values[0]
        # end = new_coord_df.Pos.values[-1]
                
        start = np.min([int(gene_coords.loc[locus, "Start"]), int(gene_coords.loc[locus, "End"])])
        end = np.max([int(gene_coords.loc[locus, "Start"]), int(gene_coords.loc[locus, "End"])])

        cropped_record = graphic_record.crop((start, end))
        cropped_record.plot(ax=ax_coords, with_ruler=False)

        ax_coords.set_xlim(start, end)
        
        # # replace 0s in all but 1 array with NaN so that they don't get plotted everywhere where there's a 0
        # plot_mean = new_coord_df.Mean.values.copy()        
        # plot_mean[plot_mean == 0] = np.nan
        
        ax_saliency.plot(X_matrix_H37Rv_coords[:, locus_idx], combined_max[:, locus_idx], color="black", linewidth=0.7)
        ax_saliency.plot(X_matrix_H37Rv_coords[:, locus_idx], combined_min[:, locus_idx], color="black", linewidth=0.7)
        
        # ax_saliency.plot(new_coord_df.Pos, new_coord_df.Max.values, color="black", linewidth=0.7)
        # ax_saliency.plot(new_coord_df.Pos, new_coord_df.Min.values, color="black", linewidth=0.7) 
        #ax_saliency.plot(new_coord_df.Pos, plot_mean, color="black", linewidth=0.7) 

        # use the largest absolute value for the given gene so that each plot is symmetric around the y axis
        #max_val = np.max(np.abs([combined_min[:, i], combined_max[:, i]]))
        #ax_saliency.set_ylim(-max_val*1.1, max_val*1.1)

        sns.despine(ax=ax_saliency, top=True, right=True, left=True, bottom=True)
        ax_coords.set_ylabel("saliency")
        
        new_coord_df = pd.DataFrame({"Gene": locus, "Pos": X_matrix_H37Rv_coords[:, locus_idx], 
                                     "Max": combined_max[:, locus_idx], "Min": combined_min[:, locus_idx],
                                     "Mean": combined_mean[i, locus_idx]})
        
        with open(os.path.join(fasta_dir, f'{locus}.fasta'), 'r') as f:
            
            # subtract 1 because this includes the newline character
            aln_len = len(f.readlines()[-1]) - 1
        
        k = 0
        for i, row in new_coord_df.iterrows():
            if pd.isnull(row["Pos"]):
                
                # include locus name to distinguish between loci in the final dataframe
                new_coord_df.loc[i, "Pos"] = f"{locus}_{k}"
                k += 1
            else:
                new_coord_df.loc[i, "Pos"] = row["Pos"]
                
#         # add correlations between MICs and saliency scores
#         new_coord_df = new_coord_df.reset_index()
#         corr_df = compute_mic_saliency_correlation_single_locus(all_scores, locus, aln_len, locus_idx, locus_list, mics)
        
#         # merge with coords dataframe
#         new_coord_df = new_coord_df.merge(corr_df, on="index", how="outer")
#         del new_coord_df["index"]
        
        #new_coord_df["Gene"] = locus
        # combine into a single dataframe
        res_df = pd.concat([res_df, new_coord_df.iloc[:aln_len, :]])
        #res_df["Pos"] = res_df["Pos"].astype(int)
       
    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join(deeplift_dir, "saliency_plots.png"), dpi=300)
    return res_df



def did_cnn_find_pos(cnn_saliency_df, drugs_lst, cat_to_check=["1", "2"]):
    
    search_df = who_variants.loc[(who_variants.drug.isin(drugs_lst)) & (who_variants.confidence.str.contains("|".join(cat_to_check)))]
    who_sites = [val for val in search_df.genome_index.str.split(",")]
    who_pos = np.unique(list(itertools.chain.from_iterable(who_sites))).astype(int)
    print(f"{len(who_pos)} Cat {'/'.join(cat_to_check)} WHO catalog sites for {drugs_lst}")

    # check if the CNN finds all sites that are Cat 1 or 2
    # non_zero_scores = cnn_saliency_df.query("Max > 0 | Min < 0")
    
    # make stricter using mean
    # non_zero_scores = cnn_saliency_df.query("Mean != 0")
    cnn_saliency_df["WHO"] = cnn_saliency_df.Pos.isin(who_pos).astype(int)
    cnn_saliency_df["Lineage_SNP"] = cnn_saliency_df.Pos.isin(freschi_snps.position).astype(int)
    non_zero_scores = cnn_saliency_df.query("Max > 0")
    
    not_found_sites = np.array(list(set(who_pos) - set(non_zero_scores.Pos)))
    if len(not_found_sites) == 0:
        print(f"CNN found all Cat {'/'.join(cat_to_check)} WHO catalog sites for {drugs_lst}")
    else:
        print(f"{len(not_found_sites)} Cat {'/'.join(cat_to_check)} WHO catalog sites not found")
        print(f"Sites not found: {np.sort(not_found_sites)}")
    return cnn_saliency_df.reset_index(drop=True)        




def bootstrap_saliencies(df, num_bootstrap=1000):
    '''
    Use only nonzero scores because there is huge zero-inflation. Sites with 0 scores are irrelevant
    '''
    
    print(f"Performing bootstrapping with {num_bootstrap} replicates...")
    
    for gene in df.Gene.unique():
        #print(f"    {gene}")
        single_df = df.query("Gene == @gene").reset_index(drop=True)
        
        bs_max_reps = []
        bs_min_reps = []
        # bs_mean_reps = []

        for _ in range(num_bootstrap):
            bs_max_idx = np.random.choice(single_df.query("Max > 0").index, len(single_df.query("Max > 0")), replace=True)        
            bs_max_reps.append(np.mean(single_df.loc[bs_max_idx, "Max"].values))

            bs_min_idx = np.random.choice(single_df.query("Min < 0").index, len(single_df.query("Min < 0")), replace=True)
            bs_min_reps.append(np.mean(single_df.loc[bs_min_idx, "Min"].values))
            
            # bs_mean_idx = np.random.choice(single_df.query("Mean != 0").index, len(single_df.query("Mean != 0")), replace=True)        
            # bs_mean_reps.append(np.mean(single_df.loc[bs_mean_idx, "Mean"].values))

        assert np.min(bs_max_reps) > 0
        assert np.max(bs_min_reps) < 0
        # assert np.mean(bs_mean_reps) != 0

        # get the proportion of scores in the resampled distributions that are at least as extreme as the test statistic for each site
        for i, row in df.query("Gene == @gene").iterrows():
            if row["Max"] > 0:
                df.loc[i, "max_pVal"] = np.mean(np.array(bs_max_reps) >= row["Max"])
            if row["Min"] < 0:
                df.loc[i, "min_pVal"] = np.mean(np.array(bs_min_reps) <= row["Min"])
        
            # if row["Mean"] < 0:
            #     non_zero_df.loc[i, "mean_pVal"] = np.mean(np.array(bs_mean_reps) <= row["Mean"])
            # else:
            #     non_zero_df.loc[i, "mean_pVal"] = np.mean(np.array(bs_mean_reps) >= row["Mean"])
                
                
    # compute global p values
    bs_max_reps = []
    bs_min_reps = []
    bs_mean_reps = []

    for _ in range(num_bootstrap):

        bs_max_idx = np.random.choice(df.query("Max > 0").index, len(df.query("Max > 0")), replace=True)        
        bs_max_reps.append(np.mean(df.loc[bs_max_idx, "Max"].values))

        bs_min_idx = np.random.choice(df.query("Min < 0").index, len(df.query("Min < 0")), replace=True)
        bs_min_reps.append(np.mean(df.loc[bs_min_idx, "Min"].values))
        
        # bs_mean_idx = np.random.choice(non_zero_df.query("Mean != 0").index, len(non_zero_df.query("Mean != 0")), replace=True)
        # bs_mean_reps.append(np.mean(non_zero_df.loc[bs_mean_idx, "Mean"].values))

        assert np.min(bs_max_reps) > 0
        assert np.max(bs_min_reps) < 0
        # assert np.mean(bs_mean_reps) != 0

    # get the proportion of scores in the resampled distributions that are at least as extreme as the test statistic for each site
    for i, row in df.iterrows():
        df.loc[i, "global_max_pVal"] = np.mean(np.array(bs_max_reps) >= row["Max"])
        df.loc[i, "global_min_pVal"] = np.mean(np.array(bs_min_reps) <= row["Min"])
        
        # if row["Mean"] < 0:
        #     non_zero_df.loc[i, "global_mean_pVal"] = np.mean(np.array(bs_mean_reps) <= row["Mean"])
        # else:
        #     non_zero_df.loc[i, "global_mean_pVal"] = np.mean(np.array(bs_mean_reps) >= row["Mean"])
            
            
            
def create_all_loci_matrices(locus_list, fasta_dir, saliency_df, df_phenos):
    '''
    Creates a dictionary of matrices with every nucleotide for every isolate in the given loci. This is to determine the variants at each site and see which are
    associated with resistance. 
    '''
    
    # store the matrices in a dictionary for easily getting the one corresponding with an argument locus
    seq_mat_all_loci = {}
    
    for locus in locus_list:
        
        seq_lst = [(seq.id, str(seq.seq)) for seq in SeqIO.parse(os.path.join(fasta_dir, f"{locus}.fasta"), "fasta")]        
        aln_len = len(seq_lst[0][1])
        seq_df = pd.DataFrame(seq_lst)
        seq_df.columns = ["Isolate", "Seq"]
        
        # fasta files contain the full VCF file name, without the .vcf extension. So use the Path column in df_phenos. 
        # The ROLLINGDB_ID column is just the isolate name, not the full file path
        seq_df = seq_df.loc[seq_df["Isolate"].isin(np.concatenate([df_phenos["Path"].values, np.array(["MT_H37Rv"])]))].reset_index(drop=True)
        
        nuc_matrix = seq_df["Seq"].str.split("", expand=True)
        nuc_matrix = nuc_matrix.iloc[:, 1:-1]

        nuc_matrix["Isolate"] = [os.path.basename(fName) for fName in seq_df["Isolate"].values]
        nuc_matrix = nuc_matrix.set_index("Isolate")
        
        # this will already be sorted. DON'T SORT AGAIN BECAUSE THEN NANS WILL BE OUT OF ORDER
        nuc_matrix.columns = saliency_df.query("Gene == @locus").Pos.values[:aln_len]
        
        seq_mat_all_loci[locus] = nuc_matrix
    
    return seq_mat_all_loci

            
            
def generate_saliency_plots(config_file, who_drugs_lst, cat_to_check=["1", "2"], binary=False, isolate_variants_df=None, num_bootstrap=10000, save=False):
        
    kwargs = yaml.safe_load(open(config_file, "r"))
    
    out_dir = kwargs["output_path"]
    fasta_dir = kwargs["genotype_input_directory"]
    drug = kwargs["drug"]
    locus_list = kwargs["locus_list"]
    df_phenos_path = kwargs["phenotype_file"]
    binary_thresh = kwargs["binary_thresh"]
        
    #fastas = glob.glob(os.path.join(out_dir, "fastas", "*.fasta"))
    fastas = [os.path.join(fasta_dir, gene + ".fasta") for gene in locus_list]
    print(f"{len(fastas)} loci!")
    
    # make the genetic coordinates dataframe. Includes strand sense and locus length
    gene_coords, sense_dict = get_gene_coords(locus_list, fasta_dir)

    # saliency_df = multi_locus_saliency(drug, out_dir, fasta_dir, locus_list, sense_dict, gene_coords, df_phenos_path, binary_thresh, binary=binary, isolate_variants_df=isolate_variants_df, save=save)
    
    saliency_df = multi_locus_saliency(config_file, sense_dict, gene_coords, binary=binary, isolate_variants_df=isolate_variants_df, save=save)
    
    # update with WHO and lineage SNP annotations (crude because only the positions are checked, not the actual SNPs)
    saliency_df = did_cnn_find_pos(saliency_df, who_drugs_lst, cat_to_check)
    
    df_phenos = pd.read_csv(df_phenos_path)
    seq_mat_all_loci = create_all_loci_matrices(locus_list, fasta_dir, saliency_df, df_phenos)
    
    # add p-values with non-parametric boostrapping
    bootstrap_saliencies(saliency_df, num_bootstrap=num_bootstrap)
    
    return saliency_df, seq_mat_all_loci