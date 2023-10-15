import sys, glob, os, vcf, tracemalloc, pickle
import numpy as np
import pandas as pd
from Bio import SeqIO
sys.path.append("utils")
from inSilicoMut_utils import *

# starting the memory monitoring
tracemalloc.start()


#################################### STEP 0: READ IN FILES AND INITIALIZE VARIABLES ####################################
    
    
######## IMPORTANT: START is 0-indexed, END is 1-indexed to be consistent with the previous SNP concatenator in Perl ########
        

if len(sys.argv) == 7:
    _, PATHS_FILE, START, END, SENSE, OUT_FILE, SAVE_FASTA = sys.argv
    ADDITIONAL_ISOLATES_FILE = None
    INSILICO_MUTS_FILE = None
    
elif len(sys.argv) == 8:
    _, PATHS_FILE, ADDITIONAL_ISOLATES_FILE, START, END, SENSE, OUT_FILE, SAVE_FASTA  = sys.argv
    INSILICO_MUTS_FILE = None
    
elif len(sys.argv) == 9:
    _, PATHS_FILE, ADDITIONAL_ISOLATES_FILE, INSILICO_MUTS_FILE, START, END, SENSE, OUT_FILE, SAVE_FASTA  = sys.argv
    
else:
    raise ValueError(f"Must pass in 7-9 command line arguments. You passed in {len(sys.argv)-1}")

START = int(START)
END = int(END)

SENSE = SENSE.upper()
assert SENSE in ["POS", "NEG"]

SAVE_FASTA = SAVE_FASTA.upper()
assert SAVE_FASTA in ["TRUE", "FALSE"]

if not os.path.isfile(PATHS_FILE):
    raise ValueError(f"{PATHS_FILE} is not a file!")
    
# PATHS_FILE should be a text file of paths
if PATHS_FILE[-4:] != ".txt":
    raise ValueError(f"{PATHS_FILE} must be a text file!")

add_paths = []

if ADDITIONAL_ISOLATES_FILE is not None:
    
    if not os.path.isfile(ADDITIONAL_ISOLATES_FILE):
        raise ValueError(f"{ADDITIONAL_ISOLATES_FILE} is not a file!")
    
    # PATHS_FILE should be a text file of paths
    if ADDITIONAL_ISOLATES_FILE[-4:] != ".txt":
        raise ValueError(f"{ADDITIONAL_ISOLATES_FILE} must be a text file!")
    
    add_paths += list(pd.read_csv(ADDITIONAL_ISOLATES_FILE, sep="\t", header=None)[0].values)

if INSILICO_MUTS_FILE is not None:

    if not os.path.isfile(INSILICO_MUTS_FILE):
        raise ValueError(f"{INSILICO_MUTS_FILE} is not a file!")
    
    # PATHS_FILE should be a text file of paths
    if INSILICO_MUTS_FILE[-4:] != ".txt":
        raise ValueError(f"{INSILICO_MUTS_FILE} must be a text file!")
    
    add_paths += list(pd.read_csv(INSILICO_MUTS_FILE, sep="\t", header=None)[0].values)

paths = pd.read_csv(PATHS_FILE, sep="\t", header=None)[0].values
print(f"Making multiple sequence alignment for {len(paths)} sequences and {len(add_paths)} additional sequences")
paths = np.concatenate([paths, np.array(add_paths)], axis=0)

if ".fasta" not in OUT_FILE:
    OUT_FILE = OUT_FILE.split(".")[0] + ".fasta"
    
if not os.path.isdir(os.path.dirname(OUT_FILE)):
    os.makedirs(os.path.dirname(OUT_FILE))
    
# H37Rv reference strain
h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
genome_len = len(h37Rv)
print(f"Reference genome size: {genome_len}")

# get only the region of interest
h37Rv_region = list(str(h37Rv.seq[START:END]))
print(f"Unaligned region size: {len(h37Rv_region)}")
del h37Rv

    
    
def allele_category(record, qualThresh=10, heteroThresh=0.25):
    '''
    Returns "alt" or "ref" if the variant is low-quality or ambiguous. Otherwise this function returns "missing"
    
    Low-quality criteria:
    
        1. FILTER == Del, LowCov
        2. FILTER == Amb and 0.1 < AF < 0.9
        3. SNP quality < 10

    Criteria for not confident in a variant or can not be reliably inserted, so leave it as reference:

        1. IMPRECISE variant (in the INFO field)
        2. Indels longer than 15 bp where neither the REF nor the ALT are of length 1 (this case is handled in the next function)
        
    If FILTER contains Amb and the alternative allele fraction > 0.9, then it is a pure alternative call. 
    If FILTER contains Amb and the alternative allele fraction < 0.1, the it is a pure reference call. 
    '''
    
    # fill in things that might be missing
    if "AF" not in record.INFO.keys():
        af = 0.76
    else:
        af_lst = record.INFO["AF"]
        
        # more than one alternative allele -- heterogeneous alternative allele
        if len(af_lst) > 1:
            return "missing"
        else:
            af = float(af_lst[0])

    # QUAL field considers read depth, base quality, mapping quality. But it is also on the Phred scale
    if record.QUAL is None:
        qual = 11
    else:
        qual = record.QUAL

    # don't include IMPRECISE variants because they are difficult to reliably impute and often aren't reliable calls anyway
    # unreliability can be due to ambiguous alignments, complex genomic regions, low sequencing coverage, assembly gaps, or segmental duplications
    # basically these are breakpoints that the variant caller is not confident in. If we put Ns, often we get huge runs of Ns, which causes too much noise for the model.

    # pilon was not able to resolve the variants (usually due to large deletions), so leave as reference
    # Del tag is for deletions, meaning the variant is in a region that is covered by a deletion in another variant
    # so don't change that variant. It will get changed to - for the deletion, then don't make any further updates
    if "IMPRECISE" in record.INFO.keys() or "Del" in record.FILTER:
        return "ref"
    
    # the filter field is an empty list of it is PASS, else the list is non-empty
    # only consider the non-Amb cases here. Amb cases will be later, check the AF too for that
    if len(record.FILTER) > 0 and "Amb" not in record.FILTER:
        return "missing"
    
    # because IMPRECISE is taken care of above, this should only return missing for cases where REF = N or ALT = N
    if "N" in record.REF or "N" in "".join(np.array(record.ALT).astype(str)):
        return "missing"
    
    # check if there are any non alphanumeric characters. This would indicate a heterogeneous alternative allele
    if not "".join(np.array(record.ALT).astype(str)).isalnum():
        return "missing"
        
    # PASS or Amb filters and an alternative allele fraction of less than 0.9 means we have a mixture of REF and ALT
    if "Amb" in record.FILTER or len(record.FILTER) == 0:
        
        # default heteroThresh = 0.25. So alternative AF > 0.75 or AF < 0.25 to be a pure alternative call or reference call, respectively
        if heteroThresh <= af <= (1-heteroThresh):
            return "missing"
        elif af < heteroThresh:
            return "ref"
        elif af > (1-heteroThresh):
            return "alt"

    # low SNP quality
    if qual < qualThresh:
        return "missing"

    # base quality, mapping quality, and read depth (measures of certainty about a variant)
    if 'DP' in record.INFO.keys():
        if record.INFO['DP'] < 5:
            return 'missing'

    if 'MQ' in record.INFO.keys():
        if record.INFO['MQ'] == 0:
            return 'missing'

    # base quality is 0 for indels, so include this step for only SNPs and MNPs (lengths are the same for REF and ALT)
    if len(record.REF) == len("".join(np.array(record.ALT).astype(str))) and 'BQ' in record.INFO.keys():
        if record.INFO['BQ'] < 20:
            return 'missing'
        
    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF < 0.25
    return "alt"



def introduce_snps_indels_single_seq(fName, h37Rv_region, START, END):
    
    new_seq = h37Rv_region.copy()

    vcf_file = vcf.Reader(filename=fName)

    # start is 0-indexed (exclusive) and end is 1-indexed (inclusive)
    for record in vcf_file:

        # get only the region of interest
        if record.POS > START and record.POS <= END:

            # get the allele type: ref, alt, or missing
            single_allele_type = allele_category(record) 

            # only change the sequence if the type is not reference
            if single_allele_type != "ref":
            
                # convert alternative allele from list to string
                alt_allele = "".join(np.array(record.ALT).astype(str))
                ref_allele = str(record.REF)
    
                # the index to replace -- this is 0-indexed, consistent with Python
                idx = record.POS - (START + 1)
    
                # no length change -- SNP or MNP. Python will replace all elements if the original and new are the same length
                if len(ref_allele) == len(alt_allele):
    
                    if single_allele_type == "alt":
                        new_seq[idx:idx+len(ref_allele)] = alt_allele
                    elif single_allele_type == "missing":
                        new_seq[idx:idx+len(ref_allele)] = ["N"]*len(alt_allele)
                    # the only other option is reference, so don't do anything
                    else:
                        continue
    
                # indels
                else:
                    
                    # insertion -- insert both alternative and missing insertions
                    if len(alt_allele) > len(ref_allele):
    
                        # replace the nucleotide at the reference index with the alternative nucleotides
                        # also add the number of gap characters needed (len(ALT) - len(REF) to insertion_dict at the appropriate index                        
                        # only input short insertions and also if they pass the QC filters. Leave the others as reference
                        if (len(alt_allele) - len(ref_allele) <= 15):
    
                            if single_allele_type == "missing":
                                alt_allele = "N"*len(alt_allele)
    
                            # if REF > 1, then the entire REF allele must be removed (across all positions) and replaced with the ALT allele
                            # do this with a dummy character, X, which will be later removed. 
                            # This is generalizable to even the case where REF == 1 because it will just replace the first index
    
                            # replace everything with X first
                            new_seq[idx:idx+len(ref_allele)] = ["X"] * len(ref_allele)
    
                            # then make the first position the alternative allele
                            new_seq[idx] = alt_allele
    
                            insertion_dict[idx] = np.max([insertion_dict[idx], len(alt_allele) - len(ref_allele)])
    
                        # don't do anything if indels are very long
                        else:
                            continue
    
                    # deletion -- insert both alternative and missing deletions IF ALT IS OF LENGTH 1
                    # insert only alternative deletions if they are <= 15 bp. Missing deletions can't be reliably inserted because you don't know where to start
                    # for the alternative case, you can match up the starts of the REF and ALT to figure out where to add gap characters
                    else:
    
                        if len(alt_allele) == 1:
    
                            new_allele = []
                            assert alt_allele in ref_allele
    
                            # iterate through the reference to find where the alternative allele comes up first, then make everything else the gap character
                            # boolean to check if we have found the alt_allele (assume that it would be the first instance of that nucleotide in the ref_allele)
                            found_alt_allele = False
    
                            for i, nuc in enumerate(ref_allele):
                                if nuc == alt_allele:
                                    if not found_alt_allele:
                                        new_allele.append(alt_allele)
                                        found_alt_allele = True
                                    else:
                                        new_allele.append("-")
                                else:
                                    new_allele.append("-")
    
                            assert len(new_allele) == len(ref_allele)
    
                            # Python will replace all elements if the original and replace string are the same length
                            # add this step so that if the allele extends more than the region of interest, it is truncated
                            old_len = len(new_seq)
                            new_seq[idx:idx+len(ref_allele)] = new_allele
                            new_seq = new_seq[:old_len]
    
                        else:                        
                            # only input short deletions and also if they pass the QC filters and if the first nucleotide of the REF and ALT are the same. 
                            # In that case, replace the remaining characters of the REF list with the ALT nucleotides
                            # the point of this is mainly for the insilico mutations. Some of them have differing lengths, but the net change is a deletion
                            if single_allele_type == "alt" and alt_allele[0] == ref_allele[0] and (len(ref_allele) - len(alt_allele) <= 15):
    
                                # the replacement is the alternative allele padded with gap characters. # of gap characters = the length difference between them 
                                new_allele = list(alt_allele) + ['-'] * (len(ref_allele) - len(alt_allele))
                                assert len(new_allele) == len(ref_allele)
    
                                # Python will replace all elements if the original and replace string are the same length
                                # add this step so that if the allele extends more than the region of interest, it is truncated
                                old_len = len(new_seq)
                                new_seq[idx:idx+len(ref_allele)] = new_allele
                                new_seq = new_seq[:old_len]
    
                            # don't put in the deletion if it is long, or the first nucleotides of REF and ALT don't match because it can not be reliably inserted
                            else:
                                continue
                        
    # check lengths because both of them are lists right now 
    assert len(new_seq) == len(h37Rv_region)
    return new_seq


#################################### STEP 1: GET SNPS AND INDELS AND INSERT INTO EACH SEQUENCE USING THE FUNCTION ABOVE ####################################


# keep track of positions and the numbers of insertions relative to H37Rv
# after this main loop, these need to be introduced into h37Rv_region and also into 
global insertion_dict
insertion_dict = dict(zip(np.arange(0, END-START), np.zeros(END-START)))

seq_dict = {}

for i, fName in enumerate(paths):
    
    seq_dict[os.path.basename(fName).replace(".eff", "").replace(".vcf", "")] = introduce_snps_indels_single_seq(fName, h37Rv_region, START, END)

    if i % 1000 == 0:
        print(i)

print(f"Finished reading {len(seq_dict)} sequences!")

# convert to dataframe for easy querying. Convert everything to integers and keep only indices where gap characters need to be inserted (num_insertion > 0)
insertion_sites = pd.DataFrame(insertion_dict, index=[0]).T.reset_index().rename(columns={"index": "aln_idx", 0:"len_insertion"})
insertion_sites = insertion_sites.query("len_insertion > 0").reset_index(drop=True)
insertion_sites[insertion_sites.columns] = insertion_sites[insertion_sites.columns].astype(int)

insertion_sites_fName = os.path.join(os.path.dirname(OUT_FILE), f"{os.path.basename(OUT_FILE).split('.')[0]}_insertion_sites.csv")
print(f"Saving dataframe of sites with insertions to {insertion_sites_fName}")
insertion_sites.to_csv(insertion_sites_fName, index=False)
    

#################################### STEP 2: FILL IN GAP CHARACTERS IN THE REFERENCE SEQUENCE ####################################

    
new_ref_seq = h37Rv_region.copy()

for _, row in insertion_sites.iterrows():
    
    # number of gap characters to add
    add_gap = row["len_insertion"]

    new_ref_seq[row["aln_idx"]] = new_ref_seq[row["aln_idx"]] + "-" * add_gap
        
# get the reverse complement if negative sense. This function returns the joined sequence. If not, 
if SENSE == "NEG":
    new_ref_seq = reverse_complement(new_ref_seq)
else:
    new_ref_seq = "".join(new_ref_seq)

print(f"Aligned region size: {len(new_ref_seq)}")
    
    
#################################### STEP 3: FILL IN GAP CHARACTERS IN ALL SEQUENCES AND WRITE TO THE OUTPUT FILE ####################################


if SAVE_FASTA == "TRUE":
    
    with open(OUT_FILE, "w+") as file:
    
        for isolate, seq in seq_dict.items():
    
            assert len(seq) == (END - START)
    
            for _, row in insertion_sites.iterrows():
    
                # the numbers in insertion_sites have 1 subtracted from them, so they are the number of nucleotides to insert
                # this is the length that that position should be
                pos_length = row["len_insertion"] + 1
    
                # compare the lengths of the nucleotides at the given index and the pos_length (with gap characters)
                if len(seq[row["aln_idx"]]) < pos_length:
                    seq[row["aln_idx"]] = seq[row["aln_idx"]] + "-" * int(pos_length - len(seq[row["aln_idx"]]))
    
                assert len(seq[row["aln_idx"]]) == pos_length
    
            # remove X characters, which are used for some insertions
            # check that the new length matches with the reference sequence, which has already had gap characters inserted
            seq = "".join(seq).replace("X", "")
            assert len(seq) == len(new_ref_seq)
    
            # get the reverse complement if negative sense
            if SENSE == "NEG":
                seq = reverse_complement(seq)
            
            # write the new sequence to the alignment file
            file.write(">" + isolate + "\n")
            file.write(seq + "\n")
        
        # write the reference sequence 
        file.write(">MT_H37Rv\n")
        file.write(new_ref_seq + "\n")
        

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")