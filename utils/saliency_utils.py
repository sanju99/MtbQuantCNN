from dna_features_viewer import BiopythonTranslator, GraphicFeature, GraphicRecord
from dna_features_viewer.biotools import annotate_biopython_record
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, sys
from Bio import SeqIO
from Bio.Seq import Seq

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "model"))
import scipy.stats as st


# import reference files
who_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/WHO_resistance_variants_all.csv")

coll_2014 = pd.read_csv("/home/sak0914/who-analysis/data/coll2014_SNP_scheme.tsv", sep="\t")
freschi_2020 = pd.read_csv("/home/sak0914/who-analysis/data/freschi2020_SNP_scheme.tsv", sep="\t")

coll_2014["position"] = coll_2014["position"].astype(int)
freschi_2020["position"] = freschi_2020["position"].astype(int)


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
record = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
print(len(record.seq))

graphic_record = MyCustomTranslator().translate_record(record)

############ CUSTOM FUNCTIONS TO GENERATE SALIENCY PLOTS ##################

def get_gene_coords(locus_list, fasta_dir):
    '''
    Use this function to get a dataframe of coordinates from the bash scripts used to generate the alignment FASTA files for every locus.
    
    coords_lst is a list of tuples of the start and end coordinates of the genes
    '''
    coords = []
    sense_lst = []
    
    for i, locus in enumerate(locus_list):
        
        # read the coordinates from the file
        with open(os.path.join(fasta_dir, locus + ".sh"), "r") as file:
            
            for line in file:
                if line[0] not in ["#", "\n"] and "make_MSA" in line:

                    # the 2nd, 3rd, and 4th to last strings are start, end, and sense
                    split_line = line.split(" ")[-4:-1]
                    coords.append([int(split_line[0]), 
                                   int(split_line[1])
                                  ])

                    # sense comes after the coordinates. Also remove any quotes that might be in the string
                    sense_lst.append(split_line[2].lower().replace('"', '').replace("'", ""))

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



def compute_saliency_score_significance(locus_idx, locus, scores_max, scores_min, permute_max_lst, permute_min_lst, sig_thresh):
    
    # get the max and min scores from the permutation tests for the locus of interest
    permute_max = [pd.Series(np.load(score_path)[:, locus_idx]) for score_path in permute_max_lst]
    permute_min = [pd.Series(np.load(score_path)[:, locus_idx]) for score_path in permute_min_lst]
    
    print(f"Plotting significant saliency scores for {locus} using {len(permute_max)} max scores and {len(permute_min)} min scores")

    permute_max = pd.concat(permute_max, axis=1).values
    permute_min = pd.concat(permute_min, axis=1).values
    
    # get the max and min scores for the main model for the locus of interest
    scores_max_locus = scores_max[:, locus_idx]
    scores_min_locus = scores_min[:, locus_idx]
    
    max_pvals = []
    min_pvals = []

    for i in range(len(scores_max_locus)):

        if scores_max_locus[i] > 0:
            max_pvals.append(np.mean(permute_max[i, :] > scores_max_locus[i]))
        else:
            max_pvals.append(np.nan)

        if scores_min_locus[i] < 0:
            min_pvals.append(np.mean(permute_min[i, :] < scores_min_locus[i]))
        else:
            min_pvals.append(np.nan)

    # return arrays of 1s and 0s, where 1 = significant
    return (np.array(max_pvals) < sig_thresh).astype(int), (np.array(min_pvals) < sig_thresh).astype(int)
            
    

def multi_locus_saliency(out_dir, binary, locus_list, sense_dict, gene_coords, fasta_dir, save=False, significance=True, sig_thresh=0.05, suffix=""):
    
    # this is 1-indexed and in reverse order for negative sense genes
    X_matrix_H37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)
    
    if binary:
        saliency_dir = os.path.join(out_dir, "saliency", "binary")
    else:
        saliency_dir = os.path.join(out_dir, "saliency", "quant")
    
    # combined_mean = np.load(os.path.join(saliency_dir, "scores_mean.npy"))
    combined_max = np.load(os.path.join(saliency_dir, f"scores_max{suffix}.npy"))
    combined_min = np.load(os.path.join(saliency_dir, f"scores_min{suffix}.npy"))
    
    # check signs
    assert np.max(combined_min.flatten()) <= 0
    assert np.min(combined_max.flatten()) >= 0
    
    # get results from the permutation test
    if significance:
        permute_max_lst = glob.glob(os.path.join(saliency_dir, f"permutation_test/scores_max*{suffix}.npy"))
        permute_min_lst = glob.glob(os.path.join(saliency_dir, f"permutation_test/scores_min*{suffix}.npy"))

    # initalize empty dataframe for saliency scores
    res_df = pd.DataFrame(columns=["Gene", "Pos", "Max", "Min"])

    fig, ax = plt.subplots(gene_coords.shape[0]*2, 1, figsize=(14, 3*gene_coords.shape[0]))
    axes=ax.flatten()

    point_size = 3

    for locus_idx, locus in enumerate(locus_list):

        # compute p-values for the max and min scores for every position
        if significance:
            max_sig, min_sig = compute_saliency_score_significance(locus_idx, locus, combined_max, combined_min, permute_max_lst, permute_min_lst, sig_thresh)
        
            max_significant_df = pd.DataFrame({"Pos": X_matrix_H37Rv_coords[:, locus_idx], 
                                           "max_score": combined_max[:, locus_idx],
                                           "max_significant": max_sig,
                                      })#.query("max_significant == 1 & max_score > 0")

            min_significant_df = pd.DataFrame({"Pos": X_matrix_H37Rv_coords[:, locus_idx], 
                                           "min_score": combined_min[:, locus_idx],
                                           "min_significant": min_sig,
                                      })#.query("min_significant == 1 & min_score < 0")

            # check that there are no significant scores of 0
            assert len(max_significant_df.query("max_significant == 1 & max_score == 0")) == 0
            assert len(min_significant_df.query("min_significant == 1 & min_score == 0")) == 0

            # replace insignificant scores with 0 so that they don't get plotted
            max_significant_df.loc[max_significant_df["max_significant"] == 0, "max_score"] = 0
            min_significant_df.loc[min_significant_df["min_significant"] == 0, "min_score"] = 0

        
        ax_coords = axes[(locus_idx)*2+1]
        ax_saliency = axes[(locus_idx)*2]

        start = np.min([int(gene_coords.loc[locus, "Start"]), int(gene_coords.loc[locus, "End"])])
        end = np.max([int(gene_coords.loc[locus, "Start"]), int(gene_coords.loc[locus, "End"])])

        cropped_record = graphic_record.crop((start, end))
        cropped_record.plot(ax=ax_coords, with_ruler=False)

        ax_coords.set_xlim(start, end)

        new_coord_df = pd.DataFrame({"Gene": locus, 
                                     "Pos": X_matrix_H37Rv_coords[:, locus_idx], 
                                     "Max": combined_max[:, locus_idx], 
                                     "Min": combined_min[:, locus_idx],
                                     #"Mean": combined_mean[:, locus_idx]
                                   })

        # the line plot should only include significant scores
        if significance:

            ax_saliency.plot(max_significant_df["Pos"], max_significant_df["max_score"], linewidth=0.7, color="black")
            ax_saliency.plot(min_significant_df["Pos"], min_significant_df["min_score"], linewidth=0.7, color="black")

            new_coord_df["Max_Sig"] = max_sig
            new_coord_df["Min_Sig"] = min_sig

        # if not significance, plot all scores
        else:
            ax_saliency.plot(X_matrix_H37Rv_coords[:, locus_idx], combined_max[:, locus_idx], linewidth=0.7, color="black")
            ax_saliency.plot(X_matrix_H37Rv_coords[:, locus_idx], combined_min[:, locus_idx], linewidth=0.7, color="black")

        sns.despine(ax=ax_saliency, top=True, right=True, left=True, bottom=True)
        ax_coords.set_ylabel("saliency")
        
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

        res_df = pd.concat([res_df, new_coord_df.iloc[:aln_len, :]])
       
    if not save:
        plt.show()
    else:
        plt.savefig(os.path.join(saliency_dir, "saliency_plots.png"), dpi=300)
    
    return res_df



def did_cnn_find_pos(cnn_saliency_df, drug, cat_to_check=["1", "2"], significance=True):
    
    search_df = who_variants.loc[(who_variants.drug == drug) & (who_variants.confidence.str.contains("|".join(cat_to_check)))]
    who_sites = [val for val in search_df.genome_index.str.split(",")]
    who_pos = np.unique(list(itertools.chain.from_iterable(who_sites))).astype(int)
    print(f"{len(who_pos)} Cat {'/'.join(cat_to_check)} WHO catalog sites for {drug}")

    # check if the CNN finds all sites as significant of the specified categories -- only RESISTANCE associated, so check max scores
    cnn_saliency_df["WHO"] = cnn_saliency_df.Pos.isin(who_pos).astype(int)
    
    # add lineage SNP designations
    cnn_saliency_df["Coll_2014"] = cnn_saliency_df.Pos.isin(coll_2014.position).astype(int)
    cnn_saliency_df["Freschi_2020"] = cnn_saliency_df.Pos.isin(freschi_2020.position).astype(int)
    
    non_zero_scores = cnn_saliency_df.query("Max > 0")
    
    if significance:
        non_zero_scores.query("Max_Sig == 1")
    
    not_found_sites = np.array(list(set(who_pos) - set(non_zero_scores.Pos)))
    if len(not_found_sites) == 0:
        print(f"CNN found all Cat {'/'.join(cat_to_check)} WHO catalog sites for {drug}")
    else:
        print(f"{len(not_found_sites)} Cat {'/'.join(cat_to_check)} WHO catalog sites not found")
        print(f"Sites not found: {np.sort(not_found_sites)}")
    return cnn_saliency_df.reset_index(drop=True)        



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
        seq_df["Isolate"] = [isolate.split(".")[0] for isolate in seq_df["Isolate"].values]
        seq_df = seq_df.set_index("Isolate")
        seq_df = seq_df.loc[np.concatenate([df_phenos["ROLLINGDB_ID"].values, np.array(["MT_H37Rv"])])]

        nuc_matrix = seq_df["Seq"].str.split("", expand=True)
        nuc_matrix = nuc_matrix.iloc[:, 1:-1]
        
        # this will already be sorted. DON'T SORT AGAIN BECAUSE THEN NANS WILL BE OUT OF ORDER
        nuc_matrix.columns = saliency_df.query("Gene == @locus").Pos.values[:aln_len]
        
        seq_mat_all_loci[locus] = nuc_matrix
    
    return seq_mat_all_loci

            
            
def generate_saliency_plots(drug, out_dir, locus_list, fasta_dir="/n/data1/hms/dbmi/farhat/Sanjana/CNN_results/fastas", cat_to_check=["1", "2"], binary=False, save=False, significance=True, sig_thresh=0.05, suffix=""):
                    
    fastas = [os.path.join(fasta_dir, gene + ".fasta") for gene in locus_list]
    print(f"{len(fastas)} loci!")
    
    # make the genetic coordinates dataframe. Includes strand sense and locus length
    gene_coords, sense_dict = get_gene_coords(locus_list, fasta_dir)
    saliency_df = multi_locus_saliency(out_dir, binary, locus_list, sense_dict, gene_coords, fasta_dir, save, significance, sig_thresh, suffix)
    
    # update with WHO and lineage SNP annotations (crude because only the positions are checked, not the actual SNPs)
    saliency_df = did_cnn_find_pos(saliency_df, drug, cat_to_check, significance)
    
    df_phenos = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/data_for_model.csv")
    seq_mat_all_loci = create_all_loci_matrices(locus_list, fasta_dir, saliency_df, df_phenos)
    
    return saliency_df, seq_mat_all_loci