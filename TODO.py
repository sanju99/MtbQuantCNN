



def get_codon_aa_change(drug, search_df, codon_start, search_AA, binary_thresh=0.5):
    
    ref_codon = Bio.SeqUtils.IUPACData.protein_letters_1to3[h37Rv.seq[codon_start-1:codon_start+2].translate()]
    
    for _, row in search_df.iterrows():

        sample_id = row["ROLLINGDB_ID"]
        vcf_reader = vcf.Reader(filename=f"/n/scratch3/users/s/sak0914/annotated_VCF/{sample_id}.eff.vcf")

        found_records = []
        new_codon = list(h37Rv.seq[codon_start-1:codon_start+2])
        for record in vcf_reader:
            if record.POS >= codon_start and record.POS <= codon_start+2:
                found_records.append(record)

        for record in found_records:
            if record.POS == codon_start:
                new_codon[0] = str(record.ALT[0])
            elif record.POS == codon_start+1:
                new_codon[1] = str(record.ALT[0])
            elif record.POS == codon_start+2:
                new_codon[2] = str(record.ALT[0])
          
        aa = Bio.SeqUtils.IUPACData.protein_letters_1to3[Seq("".join(new_codon)).translate()]
        
        # search for AAs that are not Category 1 mutations and have low MIC
        # (but they may have seemed like they had Cat 1 mutations and therefore got dropped by QC)
        # also exclude reference codons
        if aa not in search_AA and aa != ref_codon:
            if row[f"{drug}_midpoint"] < binary_thresh / 2:
                print(sample_id, aa, row[f"{drug}_midpoint"])
        #     else:
        #         print(sample_id, aa)
        # else:
        #     print(sample_id, aa)
        
        
RIF_df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/RIF/data_with_paths.csv")
df_rif = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/RIF/data_for_model.csv")


get_codon_aa_change("RIF", RIF_df.query("ROLLINGDB_ID not in @df_rif.ROLLINGDB_ID.values"), 761154, ["Leu", "Phe"])
get_codon_aa_change("RIF", RIF_df.query("ROLLINGDB_ID not in @df_rif.ROLLINGDB_ID.values"), 761139, ["Cys", "Ser"])