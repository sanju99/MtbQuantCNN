import numpy as np
import pandas as pd
import glob, os, yaml, sys

_, drug, phenos_file = sys.argv

freschi_snps = pd.read_excel("/home/sak0914/lasso/Freschi_SNPs.xlsx")
freschi_snps[["REF", "ALT"]] = freschi_snps["allele_change"].str.split("/", expand=True)

df_phenos = pd.read_csv(phenos_file)
isolate_variants = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/MOXI/isolate_variants.csv")

def get_lineage_snp_table(isolate_variants, df_phenos, freschi_snps):
    
    lineage_snp_pos = freschi_snps.position.values.astype(int)
    
    lineage_snp_sites = isolate_variants.query("POS in @lineage_snp_pos & Isolate in @df_phenos.ROLLINGDB_ID")
    lineage_snp_sites = lineage_snp_sites.merge(freschi_snps, left_on="POS", right_on="position", how="inner")
    
    lineage_snp_sites.loc[(lineage_snp_sites["REF_x"] == lineage_snp_sites["REF_y"]) & 
                          (lineage_snp_sites["ALT_x"] == lineage_snp_sites["ALT_y"]) &
                          (lineage_snp_sites["QC"] == 1),
                         "Lineage_SNP"] = 1
    
    # if a lineage-defining SNP is low-quality, assign to 0.5
    lineage_snp_sites.loc[(lineage_snp_sites["REF_x"] == lineage_snp_sites["REF_y"]) & 
                          (lineage_snp_sites["ALT_x"] == lineage_snp_sites["ALT_y"]) &
                          (lineage_snp_sites["QC"] != 1),
                         "Lineage_SNP"] = 0.5

    # need to have some placeholder
    #lineage_snp_sites["Lineage_SNP"] = lineage_snp_sites["Lineage_SNP"].fillna(-1)
    
    # 1 = lineage SNP, 0 = reference, -1 = SNP at a lineage-defining site that is not a lineage-defining SNP (right position, wrong SNP)
    
    # 1 = lineage SNP, 0.5 = low-quality lineage SNP, 0 = not a lineage SNP
    snp_table = lineage_snp_sites.pivot(index="Isolate", columns="POS", values="Lineage_SNP").fillna(0)

    # add 0s for all reference
    no_variants_files = list(set(df_phenos.ROLLINGDB_ID) - set(lineage_snp_sites.Isolate))

    for isolate in no_variants_files:
        snp_table.loc[isolate, :] = np.zeros(len(snp_table.columns))

    print(np.unique(snp_table.values))
    assert snp_table.shape[0] == df_phenos.shape[0]
    return snp_table, lineage_snp_sites


snp_table, lineage_snp_sites = get_lineage_snp_table(isolate_variants, df_phenos, freschi_snps)
assert len(snp_table) == len(df_phenos)

fName = os.path.join(os.path.dirname(df_phenos), "SNP_table.csv")
snp_table.to_csv(fName)