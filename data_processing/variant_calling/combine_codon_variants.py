import numpy as np
import pandas as pd
import os, vcf, sys, itertools, argparse
from Bio import Entrez, Seq, SeqIO

h37Rv_genes_df = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")

# dataframe of the gene(s) that each position of H37Rv is in. If not a gene, then the region is NC_{num}. This was made from the mycobrowser genes dataframe above, so the same genes and noncoding regions are reflected
h37Rv_coords_to_gene = pd.read_csv("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/h37Rv_coords_to_gene.csv")
h37Rv_coords_to_gene_dict = dict(zip(h37Rv_coords_to_gene['pos'], h37Rv_coords_to_gene['region']))

h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")

# goal: combine SNPs in a VCF file that occur on the same codon so that snpEff will annotate them properly
# snpEff does not natively consider multiple variants on the same codon and will annotate them as though they occur separately

def check_is_snp(record):
    '''
    Function to check if a variant is a SNP -- not IMPRECISE and not an indel
    
    IC=0	Number of reads in pileup calling an insertion at this locus
    DC=0	Number of reads in pileup calling a deletion at this locus

    Exclude indels with this filter
    '''
    
    # check first that it is not an IMPRECISE variant or a structural variant. IMPRECISE is a place where the variant caller has difficulty resolving the variants
    # they also don't have IC or DC fields, so you will get an error in the next block
    if 'IMPRECISE' in record.INFO.keys() or 'SVTYPE' in record.INFO.keys():
        return False
    else:     

        alt_allele = "".join(np.array(record.ALT).astype(str))
        
        # check string lengths to ensure no indels and also the IC and DC flags in the INFO field
        # skip cases where REF = ALT (not a SNP)
        if len(record.REF) == len(alt_allele) and record.INFO['IC'] == 0 and record.INFO['DC'] == 0 and record.REF != alt_allele:
            return True
    
    return False
        


def get_codon_from_pos(pos, h37Rv_genes_df):
    '''
    Returns a list of the H37Rv coordinates that are in the same codon as the provided position. 

    If two genes overlap at this site, then all coordinates are returned (3-6, depending on the degree of overlap between the reading frames of the two genes).

    Sense doesn't matter in this script because all variants are stored in the positive sense direction.
    '''

    gene = h37Rv_coords_to_gene_dict[pos]

    if gene.startswith('NC'):
        return None

    # multiple genes because they overlap
    if ',' in gene:
        gene = gene.split(',')
    else:
        gene = [gene]

    codon_lst = []

    # iterate through to account for multiple genes because they overlap
    for single_gene in gene:
        if single_gene not in h37Rv_genes_df.Symbol.values:
            raise ValueError(f"{single_gene} is not in /n/data1/hms/dbmi/farhat/Sanjana/H37Rv/mycobrowser_h37rv_genes_v4.csv")
    
        else:
            start, end = h37Rv_genes_df.query("Symbol==@single_gene")[['Start', 'End']].values[0]
        
            # Extract codon coordinates based on reading frame (considering the strand)
            for k in range(start, end, 3):
        
                # return the codon in which the argument position is in
                if pos in [k, k+1, k+2]:
                    codon_lst.append([k, k+1, k+2])

    # flatten list for easy searching later (i.e. don't have to iterate through the sublists), drop duplicates, and sort
    return np.sort(np.unique(list(itertools.chain.from_iterable(codon_lst))))




def get_variants_within_codon_to_fix(vcf_file):
    
    vcf_reader = vcf.Reader(filename=vcf_file)

    # Create a list of variant positions that are in the same codon
    pos_to_fix = []

    # Iterate through the records
    all_records = list(vcf_reader)
    
    for i, record_1 in enumerate(all_records[:-1]):

        codon_coords = get_codon_from_pos(record_1.POS, h37Rv_genes_df)

        # None only for non-coding regions
        if codon_coords is not None:
            
            for record_2 in all_records[i+1:]:
    
                if record_2.POS in codon_coords:

                    # ONLY CHANGE SNPS -- see function above
                    if check_is_snp(record_1) and check_is_snp(record_2):
        
                        # keep only one of the records because the next function will fix all variants on this codon
                        pos_to_fix.append(record_1.POS)

    # deduplicate
    return np.unique(pos_to_fix)




def combine_all_variants_single_codon(pos, vcf_fName):

    combined_entry = None
    vcf_reader = vcf.Reader(filename=vcf_fName)
    
    # get all coordinate in the codon (or multiple codons, in the case of overlapping genes) to check for variants
    # for overlapping genes, the combined variant may be longer, spanning 6 nucleotides
    coords_of_codon = get_codon_from_pos(pos, h37Rv_genes_df)

    for record in vcf_reader:
        # Check if the variant's position matches any of the codon coordinates
        if record.POS in coords_of_codon:

            if check_is_snp(record):

                # Combine information from multiple variants into a single entry
                if combined_entry is None:
                    combined_entry = record
                else:
    
                    # consecutive case is easy
                    # OR IF COMBINED_ENTRY.REF AND COMBINED_ENTRY.ALT ARE LONGER THAN 1, THAT MEANS THAT THERE IS A VARIANT AT ALL THREE SITES OF THE CODON
                    if np.abs(combined_entry.POS - record.POS) == 1 or (len(combined_entry.REF) > 1 and len(combined_entry.ALT[0]) > 1):
    
                        # combine REF nucleotides together and ALT nucleotides together. REF nucleotides are strings
                        combined_entry.REF = combined_entry.REF + record.REF
        
                        # ALT nucleotides have to be converted to strings, then put back into a list
                        combined_entry.ALT = ["".join(np.array(combined_entry.ALT).astype(str)) + "".join(np.array(record.ALT).astype(str))]
    
                    # if they are not consecutive records, then need to insert reference nucleotides to the REF and ALT fields, but don't need to change any of the other fields because it's not a variant
                    else:
    
                        # get all nucleotides in the codon coordinates list that are not in either the combined_entry or current record
                        missing_coords = list(set(coords_of_codon) - set([combined_entry.POS]) - set([record.POS]))
    
                        # fill in reference nucleotides. Subtract 1 to go from 1-indexed to 0-indexed
                        missing_nucleotides = [str(h37Rv.seq)[coord - 1] for coord in missing_coords]
    
                        # keep track in a dictionary for when there are more than 3 coordinates in coords_of_codon, so it won't always just be the middle coordinate that needs to be filled in
                        fill_in_dict = dict(zip(missing_coords, missing_nucleotides))
    
                        # add in the two records being worked on currently
                        fill_in_dict[combined_entry.POS] = combined_entry.REF
                        fill_in_dict[record.POS] = record.REF
                        
                        combined_entry.REF = "".join([fill_in_dict[pos] for pos in coords_of_codon])
    
                        new_alt = ""
    
                        for pos in coords_of_codon:
                            if pos == combined_entry.POS:
                                new_alt += "".join(np.array(combined_entry.ALT).astype(str))
                            elif pos == record.POS:
                                new_alt += "".join(np.array(record.ALT).astype(str))
                            else:
                                assert pos in missing_coords
                                new_alt += fill_in_dict[pos]
    
                        combined_entry.ALT = ["".join(new_alt)]
    
                    # same for both cases, don't change if the records are consecutive or not
                    # Take the most upstream position among the variants
                    combined_entry.POS = np.min([record.POS, combined_entry.POS])
    
                    # take the mean quality
                    if combined_entry.QUAL is not None:
                        if record.QUAL is not None:
                            combined_entry.QUAL = int(np.round(np.mean([combined_entry.QUAL, record.QUAL])))
                    else:
                        if record.QUAL is not None:
                            combined_entry.QUAL = record.QUAL
    
                    # FILTER field is a list so it can't be combined. combine both values in the FILTER field with a comma (this is taken care of in later scripts). then put into a list
                    # don't need to do anything if the filters are the same
                    if record.FILTER != combined_entry.FILTER:
                        # combined entry is PASS and the new entry is not PASS, then keep the new one
                        if len(combined_entry.FILTER) == 0 and len(record.FILTER) != 0:
                            combined_entry.FILTER = record.FILTER
    
                        # don't create a new type because then you have to add it to the header field of the VCF file
                        elif len(combined_entry.FILTER) != 0 and len(record.FILTER) != 0:
    
                            # FILTER field needs to be a list
                            combined_filter = list(np.sort(np.unique(list(combined_entry.FILTER) + list(record.FILTER))))
    
                            # options are Amb, and Del, LowCov. Sort, then take the last one, so LowCov is preferentially kept over Del, which is kept over Amb
                            # The sorted version is essentially in order of decreasing quality, so keep the last one (the lowest quality indicator)
                            if len(combined_filter) > 1:
                                combined_entry.FILTER = [combined_filter[-1]]
                            else:
                                combined_entry.FILTER = combined_filter
    
                        # else, if len(combined_entry.FILTER) != 0 and len(record.FILTER) == 0, do nothing, keep the FILTER value of combined_entry         
                    
                    updated_info_field = {}
                    
                    # Iterate through INFO fields to combine them
                    for info_field, info_value in record.INFO.items():
    
                        if info_field in combined_entry.INFO.keys():
    
                            # list type: sum them because these are measuring different metrics across the 4 bases
                            if type(info_value) == list:
    
                                # convert to arrays to use numpy functions
                                info_value = np.array(info_value)
                                new_info_value = np.array(combined_entry.INFO[info_field])
    
                                # also check that there are actually multiple values. Some fields will have single numbers in a list
                                if len(info_value) > 1:
                                    
                                    # check that lengths match
                                    assert len(info_value) == len(new_info_value)
        
                                    # sum them across the axis of the four bases, but then new values MUST be a list
                                    updated_info_field[info_field] = list(info_value + new_info_value)
                                
                                # otherwise, take the mean across the values. Apply np.squeeze first because these are arrays of length 1
                                else:
                                    updated_info_field[info_field] = np.round(np.mean([np.squeeze(info_value), np.squeeze(combined_entry.INFO[info_field])]), 2)
                            
                            # otherwise, take the mean across the values. These are just floats, so you can take the mean directly
                            else:
                                # for AF, take the lowest AF because there can be two variants on the same codon, one with low AF and one with high AF
                                # the mean could be high AF, but we want to make sure it remains ambiguous
                                if info_field == 'AF':
                                    updated_info_field[info_field] = np.min([info_value, combined_entry.INFO[info_field]])
                                else:
                                    updated_info_field[info_field] = np.round(np.mean([info_value, combined_entry.INFO[info_field]]), 2)
                                    
                        else:
                            updated_info_field[info_field] = info_value
    
                    combined_entry.INFO.update(updated_info_field)
            else:
                # remove the non-SNP coordinate from coords_of_codon so that it will be an unchanged record and will be copied directly from the original VCF to the combined codons one
                coords_of_codon = coords_of_codon[coords_of_codon != record.POS]
            
    return combined_entry, coords_of_codon


parser = argparse.ArgumentParser()

# dest indicates the name that each argument is stored in so that you can access it after running .parse_args()
parser.add_argument('-i', type=str, dest='vcf_fName', help='VCF file to combine codons for', required=True)
parser.add_argument('-o', type=str, dest='updated_vcf_fName', help='Output VCF file with combined codons')

cmd_line_args = parser.parse_args()
vcf_fName = cmd_line_args.vcf_fName
updated_vcf_fName = cmd_line_args.updated_vcf_fName

if updated_vcf_fName is None:
    updated_vcf_fName = vcf_fName.replace(".vcf", '_combinedCodons.vcf')

if not os.path.isfile(updated_vcf_fName):

    # positions that need to be fixed. It's not exhaustive, just the bare minimum. i.e. if POS = 10 is in here, then all variants with a position in the same codon as 10 will be fixed if needed
    pos_to_fix = get_variants_within_codon_to_fix(vcf_fName)
    
    combined_records = []
    coords_full_lst = []
    
    for pos in pos_to_fix:
    
        combined_record, codon_coords = combine_all_variants_single_codon(pos, vcf_fName)
        combined_records.append(combined_record)
    
        # get full list of positions that were changed so that we can keep the unchanged records constant
        coords_full_lst.append(codon_coords)
    
    combined_records = list(np.unique(combined_records))
    coords_full_lst = np.unique(list(itertools.chain.from_iterable(coords_full_lst)))
    
    unchanged_records = []
    vcf_reader = vcf.Reader(filename=vcf_fName)
    
    for record in vcf_reader:
        if record.POS not in coords_full_lst:
            unchanged_records.append(record)
            
    # combine the records that were combined on the same codon and the unchanged ones into a single list
    all_records = unchanged_records + combined_records
    all_records = sorted(all_records, key=lambda record: record.POS) # sort by position
    
    vcf_writer = vcf.Writer(open(updated_vcf_fName, "w"), vcf_reader)
    
    # Write the updated records to the new VCF file
    for updated_record in all_records:
        vcf_writer.write_record(updated_record)
    
    # Close the VCF Writer
    vcf_writer.close()