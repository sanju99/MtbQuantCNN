from dna_features_viewer import BiopythonTranslator, GraphicFeature, GraphicRecord
from dna_features_viewer.biotools import annotate_biopython_record
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, yaml, sparse, itertools, sys
from Bio import SeqIO
from Bio.Seq import Seq
import scipy.stats as st

from data_utils import *
from inSilicoMut_utils import *

data_utils_dir = "./data_processing/data_utils"
model_loci = pd.read_csv(f"{data_utils_dir}/drug_loci.csv")
model_loci[['Start', 'End']] = model_loci[['Start', 'End']].astype(int)

amino_acid_biophysical_properties = pd.read_csv(f"{data_utils_dir}/biophysical_properties_AA.csv", index_col=[0])

# import reference files
h37Rv_seq = SeqIO.read(f"{data_utils_dir}/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
h37Rv_genes = pd.read_csv(f"{data_utils_dir}/H37Rv/mycobrowser_h37rv_genes_v4.csv")
h37Rv_coords_to_gene = pd.read_csv(f"{data_utils_dir}/H37Rv/h37Rv_coords_to_gene.csv.gz", compression='gzip')
h37Rv_coords_to_gene_dict = dict(zip(h37Rv_coords_to_gene['pos'], h37Rv_coords_to_gene['region']))

BASE_TO_COLUMN = {'A': 0, 'C': 1, 'T': 2, 'G': 3, '-': 4}

who_variants_V1 = pd.read_csv(f"{data_utils_dir}/WHO_catalog_V1.csv")
who_variants_V2 = pd.read_csv(f"{data_utils_dir}/WHO_catalog_V2.csv")
coll_2014 = pd.read_csv(f"{data_utils_dir}/coll2014_SNP_scheme.tsv", sep="\t")
coll_2014["position"] = coll_2014["position"].astype(int)

drug_abbr_dict = {"Delamanid": "DLM",
                  "Bedaquiline": "BDQ",
                  "Clofazimine": "CFZ",
                  "Ethionamide": "ETO",
                  "Linezolid": "LZD",
                  "Moxifloxacin": "MXF",
                  "Capreomycin": "CAP",
                  "Amikacin": "AMK",
                  "Pretomanid": "PTM",
                  "Pyrazinamide": "PZA",
                  "Kanamycin": "KAN",
                  "Levofloxacin": "LFX",
                  "Streptomycin": "STM",
                  "Ethambutol": "EMB",
                  "Isoniazid": "INH",
                  "Rifampicin": "RIF"
                 }

abbr_drug_dict = {value: key for key, value in drug_abbr_dict.items()}


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
graphic_record = MyCustomTranslator().translate_record(h37Rv_seq)


############ CUSTOM FUNCTIONS TO GENERATE SALIENCY PLOTS ##################


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
            
    

def multi_locus_saliency(drug, out_dir, locus_list, sense_dict, gene_coords, fasta_dir, save=False, significance=True, sig_thresh=0.05, suffix=""):
    
    # this is 1-indexed and in reverse order for negative sense genes
    X_matrix_H37Rv_coords = make_h37rv_coordinates(gene_coords, locus_list, fasta_dir)
    saliency_dir = os.path.join(out_dir, "saliency")
    
    # combined_mean = np.load(os.path.join(saliency_dir, "scores_mean.npy"))
    combined_max = np.load(os.path.join(saliency_dir, f"scores_max{suffix}.npy"))
    combined_min = np.load(os.path.join(saliency_dir, f"scores_min{suffix}.npy"))
    
    # check signs
    assert np.max(combined_min.flatten()) <= 0
    assert np.min(combined_max.flatten()) >= 0

    # if we're using only Tier 1 loci, but a Tier 2 locus is longer than the Tier 1 loci, need to add additional padding characters
    if X_matrix_H37Rv_coords.shape[0] < combined_max.shape[0]:

        pad_length = combined_max.shape[0] - X_matrix_H37Rv_coords.shape[0]
        pad_matrix = (np.ones((pad_length, X_matrix_H37Rv_coords.shape[1]))*np.nan)

        # pad with NaNs
        X_matrix_H37Rv_coords = np.concatenate([X_matrix_H37Rv_coords, pad_matrix], axis=0)

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

            # print(locus, locus_idx, len(X_matrix_H37Rv_coords[:, locus_idx]), len(combined_max[:, locus_idx]), len(max_sig))
            
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
        ax_saliency.set_xlim(start, end)
        
        new_coord_df = pd.DataFrame({"Gene": locus, 
                                     "Pos": X_matrix_H37Rv_coords[:, locus_idx], 
                                     "Max": combined_max[:, locus_idx], 
                                     "Min": combined_min[:, locus_idx],
                                     #"Mean": combined_mean[:, locus_idx]
                                   })

        # the line plot should only include significant scores
        if significance:

            for k in range(len(max_significant_df)):

                # only plot non-zero scores, and plot them as vertical lines to make it cleaner than a continuous line plot, where all the points are connected
                if max_significant_df["max_score"][k] > 0:
                    ax_saliency.vlines(x=max_significant_df["Pos"][k], ymin=0, ymax=max_significant_df["max_score"][k], color="black", linewidth=0.7)
                
                if min_significant_df["min_score"][k] < 0:
                    ax_saliency.vlines(x=min_significant_df["Pos"][k], ymin=min_significant_df["min_score"][k], ymax=0, color="black", linewidth=0.7)

            # ax_saliency.plot(max_significant_df["Pos"], max_significant_df["max_score"], linewidth=0.7, color="black")
            # ax_saliency.plot(min_significant_df["Pos"], min_significant_df["min_score"], linewidth=0.7, color="black")

            new_coord_df["Max_Sig"] = max_sig
            new_coord_df["Min_Sig"] = min_sig

        # if not significance, plot all scores
        else:
            # ax_saliency.plot(X_matrix_H37Rv_coords[:, locus_idx], combined_max[:, locus_idx], linewidth=0.7, color="black")
            # ax_saliency.plot(X_matrix_H37Rv_coords[:, locus_idx], combined_min[:, locus_idx], linewidth=0.7, color="black")

            for k in range(len(combined_max[:, locus_idx])):
                
                # only plot non-zero scores, and plot them as vertical lines to make it cleaner than a continuous line plot, where all the points are connected
                if combined_max[:, locus_idx][k] > 0:
                    ax_saliency.vlines(x=X_matrix_H37Rv_coords[:, locus_idx][k], ymin=0, ymax=combined_max[:, locus_idx][k], color="black", linewidth=0.7)
                
                if combined_min[:, locus_idx][k] < 0:
                    ax_saliency.vlines(x=X_matrix_H37Rv_coords[:, locus_idx][k], ymin=combined_min[:, locus_idx][k], ymax=0, color="black", linewidth=0.7)

        # plot a black line for the x-axis at y = 0
        ax_saliency.hlines(y=0, xmin=start, xmax=end, linewidth=0.7, color='black')
        sns.despine(ax=ax_saliency, top=True, right=True, left=True, bottom=True)
        ax_coords.set_ylabel("saliency")

        # get the length of the MT_H37Rv alignment, which is the last one, making sure to remove newline characters
        with open(os.path.join(fasta_dir, f'{locus}.fasta'), 'r') as file:
            aln_len = len(file.readlines()[-1].strip())
        
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
        plt.savefig(f"/home/sak0914/MtbQuantCNN/results/{abbr_drug_dict[drug]}/saliency_plots.svg")
        plt.close()
    
    return res_df



def did_cnn_find_pos(cnn_saliency_df, drug, cat_to_check=["1", "2"], significance=True):

    # drug_full_name = abbr_drug_dict[drug]
    search_df = who_variants_V1.loc[(who_variants_V1['drug'] == drug) & (who_variants_V1.confidence.str.contains("|".join(cat_to_check)))]
    # search_df = who_variants_V2.loc[(who_variants_V2['drug'] == abbr_drug_dict[drug]) & (who_variants_V2['FINAL CONFIDENCE GRADING'].str.contains("|".join(cat_to_check)))]
    who_sites = [val for val in search_df.genome_index.str.split(",")]
    who_sites = np.unique(list(itertools.chain.from_iterable(who_sites))).astype(int)
    print(f"{len(who_sites)} Cat {'/'.join(cat_to_check)} WHO catalog sites for {drug}")

    # check if the CNN finds all sites as significant of the specified categories -- only RESISTANCE associated, so check max scores
    cnn_saliency_df["WHO"] = cnn_saliency_df.Pos.isin(who_sites).astype(int)
    
    # add lineage SNP designations
    cnn_saliency_df["Coll_2014"] = cnn_saliency_df.Pos.isin(coll_2014.position).astype(int)
    
    non_zero_scores = cnn_saliency_df.query("Max > 0")
    
    if significance:
        non_zero_scores.query("Max_Sig == 1")
    
    not_found_sites = np.array(list(set(who_sites) - set(non_zero_scores.Pos)))
    if len(not_found_sites) == 0:
        print(f"CNN found all Cat {'/'.join(cat_to_check)} WHO catalog sites for {drug}")
    else:
        print(f"{len(not_found_sites)} Cat {'/'.join(cat_to_check)} WHO catalog sites not found")
        print(f"Sites not found: {np.sort(not_found_sites)}")
    return cnn_saliency_df.reset_index(drop=True)        



def create_all_loci_matrices_for_saliency(locus_list, fasta_dir, saliency_df, df_phenos):
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
        
        # fasta files contain the full VCF file name, without the .vcf extension
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
    
    
    
def generate_saliency_plots(drug, out_dir, locus_list, fasta_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs", cat_to_check=["1", "2"], save=False, significance=True, sig_thresh=0.05, suffix=""):

    fastas = [os.path.join(fasta_dir, gene + ".fasta") for gene in locus_list]
    print(f"{len(fastas)} loci!")
    
    # make the genetic coordinates dataframe. Includes strand sense and locus length
    gene_coords, sense_dict = get_gene_coords(locus_list, fasta_dir)
    saliency_df = multi_locus_saliency(drug, out_dir, locus_list, sense_dict, gene_coords, fasta_dir, save, significance, sig_thresh, suffix)
    
    # update with WHO and lineage SNP annotations (crude because only the positions are checked, not the actual SNPs)
    # saliency_df = did_cnn_find_pos(saliency_df, drug, cat_to_check, significance)

    if 'augment' in out_dir:
        df_phenos = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}_augment/data_for_model.csv").query("Span_CC==0 & category=='train_set'")
    else:
        df_phenos = pd.read_csv(f"/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/{drug}/data_for_model.csv").query("Span_CC==0 & category=='train_set'")
        
    seq_mat_all_loci = create_all_loci_matrices_for_saliency(locus_list, fasta_dir, saliency_df, df_phenos)
    
    return saliency_df, seq_mat_all_loci





def extract_saliency_score_variant_types(saliency_df, seqDict, scores_matrix, max_scores=True, genes_lst=None):
    '''
    This function extracts each position with a nonzero saliency score and associated metadata. If there are multiple alleles at a site, it populates them as separate rows
    and adds individidual saliency scores for each allele.

    This makes it easier to search for the features responsible for different saliency peaks. 
    '''
    
    variants_df = pd.DataFrame(columns=["Gene", "POS", "REF", "ALT", "REF_AA", "ALT_AA", "Type", "Score"])

    if genes_lst is None:
        genes_lst = list(seqDict.keys())
    
    for gene in genes_lst:

        print(f"Extracting saliency score variants for {gene}")

        start, end, sense = h37Rv_genes.query("Symbol==@gene")[["Start", "End", "Strand"]].values[0]
        start_idx = np.min([list(seqDict[gene].columns).index(start), list(seqDict[gene].columns).index(end)])
        end_idx = np.max([list(seqDict[gene].columns).index(start), list(seqDict[gene].columns).index(end)])
        coding_region_df = seqDict[gene].iloc[:, start_idx:end_idx + 1]

        # check both with OR logic because depending on the sense, the start could be smaller or later than the end
        assert coding_region_df.columns[0] == start or coding_region_df.columns[0] == end
        assert coding_region_df.columns[-1] == start or coding_region_df.columns[-1] == end

        for pos in coding_region_df.columns:

            pos_idx = list(seqDict[gene].columns).index(pos)
            gene_idx = list(seqDict.keys()).index(gene)
            ref_nuc = seqDict[gene].loc['MT_H37Rv', pos]
            
            if max_scores:
                scores_df = pd.DataFrame({"Nuc": BASE_TO_COLUMN.keys(), "Score": np.max(scores_matrix[:, :, pos_idx, gene_idx], axis=0)})
            else:
                scores_df = pd.DataFrame({"Nuc": BASE_TO_COLUMN.keys(), "Score": np.max(scores_matrix[:, :, pos_idx, gene_idx], axis=0)})

            # check if there are any alternative alleles with nonzero scores
            nonzero_score_df = scores_df.query("Nuc != @ref_nuc & Score != 0").reset_index(drop=True)

            if len(nonzero_score_df) > 0:

                if type(pos) == float:
            
                    # get the codon for the given nucleotide. Only search among non-indels otherwise the legnth is too long
                    codon_num = int(np.floor(list(coding_region_df.columns[~coding_region_df.columns.astype(str).str.contains("_")]).index(pos) / 3)) + 1
                    codon, codon_pos = get_codon_from_seq(h37Rv_seq.seq, codon_num, start, end, sense)
                    
                    # check that we got the correct codon and positions
                    assert pos in codon_pos
                    codon_nuc_dict = dict(zip(codon_pos, codon))
            
                    for i, row in nonzero_score_df.iterrows():

                        # these are deletions because the alternative allele is -, not the reference
                        if row["Nuc"] == '-':
                            variants_df = pd.concat([variants_df, pd.DataFrame({"Gene": gene, "POS": str(int(pos)), "REF": ref_nuc, "ALT": row["Nuc"],
                                                                               "REF_AA": "", "ALT_AA": "", "Type": "del", "Score": row["Score"]}, 
                                                                               index=[0])])
                        else:
                            codon_nuc_dict[pos] = row["Nuc"]
                            new_codon = "".join(codon_nuc_dict.values())
                
                            # see if there is a synonymous mutation among the nucleotides that have nonzero scores
                            if Seq(codon).translate() == Seq(new_codon).translate():
                                mut_type = "syn"
                            else:
                                mut_type = "mis"
                                
                            variants_df = pd.concat([variants_df, pd.DataFrame({"Gene": gene, "POS": str(int(pos)), "REF": ref_nuc, "ALT": codon_nuc_dict[pos], 
                                                                                "REF_AA": Seq(codon).translate(), "ALT_AA": Seq(new_codon).translate(), "Type": mut_type, "Score": row["Score"]}, 
                                                                               index=[0])])

                else:

                    # these are insertions because the reference allele is -, not the alternative
                    for i, row in nonzero_score_df.iterrows():
                        
                        variants_df = pd.concat([variants_df, pd.DataFrame({"Gene": gene, "POS": pos, "REF": ref_nuc, "ALT": row["Nuc"], 
                                                                            "REF_AA": "", "ALT_AA": "", "Type": "ins", "Score": row["Score"]}, 
                                                                           index=[0])])

    variants_df = variants_df.reset_index(drop=True)
    return variants_df