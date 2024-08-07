import argparse, subprocess, glob, os, vcf, tracemalloc, pickle
import numpy as np
import pandas as pd
from Bio import SeqIO

# starting the memory monitoring
tracemalloc.start()


#################################### STEP 0: READ IN FILES AND INITIALIZE VARIABLES ####################################
    
    
######## IMPORTANT: START is 0-indexed, END is 1-indexed (i.e., 0-indexed half-open) to be consistent with other bioinformatics tools ########
        

parser = argparse.ArgumentParser()

# Add a required string argument for the paths file
parser.add_argument("-f", type=str, dest='PATHS_FILE', help='Text file of VCF paths to include in alignment', required=True)

# dest indicates the name that each argument is stored in so that you can access it after running .parse_args()
parser.add_argument('-start', type=int, dest='START', help='Start coordinate for alignment (0-indexed, inclusive)', required=True)
parser.add_argument('-end', type=int, dest='END', help='End coordinate for alignment (1-indexed, exlusive)', required=True)
parser.add_argument('-sense', type=str, dest='SENSE', help='Sense, must be one of pos, neg, POS, NEG', required=True)
parser.add_argument('-o', type=str, dest='OUT_FILE', help='Name of the output FASTA file', required=True)

parser.add_argument('--g', type=str, dest="GENOME_FILE", default="/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", help="Full path to Genbank file for the reference genome")
parser.add_argument('--save-fasta', dest='SAVE_FASTA', action='store_true', help='Flag to determine if the alignment should be written')
parser.add_argument('--f2', type=str, dest='ADDITIONAL_ISOLATES_FILE', help='Text file of additional VCFs to include in alignment')
parser.add_argument('--insilico-muts-file', dest='INSILICO_MUTS_FILE', type=str, help='Text file of insilico mutations to include in alignment')
parser.add_argument('--AF_thresh', type=float, dest='AF_THRESH', default=0.75, help='Alternative allele frequency threshold (exclusive) to consider variants present')

cmd_line_args = parser.parse_args()

# required arguments
PATHS_FILE = cmd_line_args.PATHS_FILE
START = cmd_line_args.START
END = cmd_line_args.END
SENSE = cmd_line_args.SENSE
OUT_FILE = cmd_line_args.OUT_FILE
GENOME_FILE = cmd_line_args.GENOME_FILE

# optional arguments
SAVE_FASTA = cmd_line_args.SAVE_FASTA
ADDITIONAL_ISOLATES_FILE = cmd_line_args.ADDITIONAL_ISOLATES_FILE
INSILICO_MUTS_FILE = cmd_line_args.INSILICO_MUTS_FILE
AF_thresh = cmd_line_args.AF_THRESH # variants with an AF > AF_thresh are considered present. AF ≤ 1 - AF_thresh is absent. Everything else is missing

if START >= END:
    raise ValueError(f"START coordinate must be less than END coordinate. You passed in START = {START} and END = {END}")

SENSE = SENSE.upper()

if SENSE not in ["POS", "NEG"]:
    raise ValueError(f"SENSE argument must be one of pos, neg, POS, NEG. You passed in {SENSE}")

# must be a float less than or equal to 1
if AF_thresh > 1:
    AF_thresh /= 100
    
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
print(f"Concatenating SNPs/indels for protein alignment for {len(paths)} sequences and {len(add_paths)} additional sequences")
paths = np.concatenate([paths, np.array(add_paths)], axis=0)

if ".fasta" not in OUT_FILE:
    OUT_FILE = OUT_FILE.split(".")[0] + ".fasta"
    
if not os.path.isdir(os.path.dirname(OUT_FILE)):
    os.makedirs(os.path.dirname(OUT_FILE))
    
# H37Rv reference strain
h37Rv = SeqIO.read(GENOME_FILE, "genbank")
genome_len = len(h37Rv)
print(f"Reference genome size: {genome_len}")

# get only the region of interest
h37Rv_region = list(str(h37Rv.seq[START:END]))
del h37Rv
    
def reverse_complement(seq):
    
    comp_dict = {'A': 'T', 
                 'C': 'G', 
                 'G': 'C', 
                 'T': 'A', 
                 'N': 'N', 
                 '-': '-'
                }
    
    # this is to turn it into a list where each element is of length 1
    seq = list("".join(seq))
    
    if len(np.unique(seq)) > 6:
        raise ValueError(f"More than 6 types of characters in the sequence!")

    if "X" in np.unique(seq):
        raise ValueError(f"There are Xs in the sequence!")
        
    seq = [comp_dict[base] for base in seq] 
    
    # reverse the sequence and return as a list
    return "".join(seq[::-1])
    
    
    
def allele_category(record, qualThresh=10, presentThresh=0.75):
    '''
    Returns "alt" or "ref" if the variant is low-quality or ambiguous. Otherwise this function returns "missing"
    
    Low-quality criteria:
    
        1. FILTER == Del, LowCov
        2. FILTER == Amb and 0.25 < AF <=0.75
        3. SNP quality < 10

    Criteria for not confident in a variant or can not be reliably inserted, so leave it as reference:

        1. IMPRECISE variant (in the INFO field)
        2. Indels longer than 15 bp where neither the REF nor the ALT are of length 1 (this case is handled in the next function)
        
    If FILTER contains Amb and the alternative allele fraction > 0.9, then it is a pure alternative call. 
    If FILTER contains Amb and the alternative allele fraction < 0.1, the it is a pure reference call. 
    '''

    alt_allele = "".join(np.array(record.ALT).astype(str))
    
    # fill in things that might be missing
    if "AF" not in record.INFO.keys():
        af = 0.76
    else:
        af_lst = record.INFO["AF"]
        
        # more than one alternative allele, which shouldn't happen for haploid organisms
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
    # pilon was not able to resolve the variants (usually due to large deletions), so leave as reference because we don't know what the variant is with high confidence

    # this occurs if there is a deletion upstream of this variant. This variant doesn't actually exist because there is no nucleotide there
    # it has been deleted by a deletion upstream. So don't 
    if "Del" in record.FILTER and len(record.REF) == len(alt_allele):
        return "ref"
        
    if "IMPRECISE" in record.INFO.keys():
        return "ref"
    
    # the filter field is an empty list of it is PASS, else the list is non-empty
    # only consider the non-Amb cases here. Amb cases will be later, check the AF too for that
    if len(record.FILTER) > 0 and "Amb" not in record.FILTER:
        return "missing"
    
    # because IMPRECISE is taken care of above, this should only return missing for cases where REF = N or ALT = N
    if "N" in record.REF or "N" in alt_allele:
        return "missing"

    if 'LowCov' in record.FILTER:
        return "missing"
    
    # check if there are any non alphanumeric characters. This would indicate a heterogeneous alternative allele
    if not alt_allele.isalnum():
        return "missing"

    # low SNP quality
    if qual < qualThresh:
        return "missing"

    # base quality, mapping quality, and read depth (measures of certainty about a variant)
    if 'DP' in record.INFO.keys():
        if record.INFO['DP'] < 5:
            return 'missing'

    if 'MQ' in record.INFO.keys():
        if record.INFO['MQ'] < 30:
            return 'missing'

    # base quality is 0 for indels, so include this step for only SNPs and MNPs (lengths are the same for REF and ALT)
    if len(record.REF) == len(alt_allele) and 'BQ' in record.INFO.keys():
        if record.INFO['BQ'] < 20:
            return 'missing'

    # PASS or Amb filters and an alternative allele fraction of less than 0.75 means we have a mixture of REF and ALT
    if "AF" in record.INFO.keys():
        # ≤ 25%, absent
        if af <= (1 - presentThresh):
            return "ref"
        elif af > presentThresh:
            return "alt"
        else:
            return "missing"
        
    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF <= 0.25
    return "alt"



def introduce_snps_indels_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=0.75):
    
    new_seq = h37Rv_region.copy()
    
    # create the tabix file if it doesn't exist
    if not os.path.isfile(f"{fName}.bgz.tbi"):

        print(f"Creating tabix file for {fName}")

        # bgzip the VCF file
        if not os.path.isfile(f"{fName}.bgz"):
            subprocess.run(f"bgzip -c {fName} > {fName}.bgz", shell=True)

        # tabix the bgzipped file, which will create fName.bgz.tbi
        subprocess.run(f"tabix -0 -p vcf {fName}.bgz -f", shell=True)

    # VCF file was indexed using 0-indexed half-open scheme, so keep START and END coords as they are
    vcf_reader = vcf.Reader(filename=f"{fName}.bgz", compressed=True)

    # need to read in the bgzipped file in order to use fetch
    records = vcf_reader.fetch('NC_000962.3', start=START, end=END)

    # start is 0-indexed (exclusive) and end is 1-indexed (inclusive)
    for record in records:

        # convert alternative allele from list to string
        alt_allele = "".join(np.array(record.ALT).astype(str))
        ref_allele = str(record.REF)

        # the index to replace -- this is 0-indexed, consistent with Python
        idx = record.POS - (START + 1)
        
        # get the allele type: ref, alt, or missing
        single_allele_type = allele_category(record, qualThresh, presentThresh)
        
        orig_ref_len = len(ref_allele)
        orig_alt_len = len(alt_allele)
        
        # only change the sequence if the type is not reference
        if single_allele_type != "ref":

            # variant occurs upstream of the region of interest but it still affects the region of interest
            # then need to remove that many nucleotides from the REF and ALT. Will only work if len(REF) == len(ALT)
            # should work for all cases because if len(ALT) or len(REF) < abs(idx), then the allele will be the empty string
            # the length of that is 0, so it should work fine with insertions and deletions. 
            if idx < 0:

                # remove the first N nucleotides from REF and ALT if that's the case
                ref_allele = ref_allele[-idx:]
                alt_allele = alt_allele[-idx:]
                    
                # then make the index 0 because now we've shifted the record to the start of the region of interest
                idx = 0

            # if the variant extends past the region of interest, then need to truncate both ref and alt alleles to avoid length mismatches
            # need to use the original lengths because if an N-terminus truncation happened, then technically the position should have been adjusted to 0
            if record.POS + orig_ref_len > END or record.POS + orig_alt_len > END:

                # the C terminus nucleotides that need to be removed
                extra_len = np.max([record.POS + orig_ref_len - END, record.POS + orig_alt_len - END])
                
                ref_allele = ref_allele[:-extra_len]
                alt_allele = alt_allele[:-extra_len]

            # no length change -- SNP or MNP. Python will replace all elements if the original and new are the same length
            if len(ref_allele) == len(alt_allele):

                old_len = len(new_seq)

                if single_allele_type == "alt":
                    new_seq[idx:idx+len(ref_allele)] = alt_allele
                elif single_allele_type == "missing":
                    new_seq[idx:idx+len(ref_allele)] = "N" * len(alt_allele)
                # the only other option is reference, so don't do anything
                else:
                    continue

            # indels
            else:
                
                # insertion -- insert both high- and low-quality insertions
                if len(alt_allele) > len(ref_allele):
                    
                    # replace the nucleotide at the reference index with the alternative nucleotides
                    # only input low-quality insertions up to 100 bp. Leave the others as reference to avoid introducing long runs of Ns into the aln
                    if single_allele_type == "alt" or (single_allele_type == "missing" and (len(alt_allele) - len(ref_allele) <= 100)):
                            
                        if single_allele_type == "missing":
                            alt_allele = "N" * len(alt_allele)

                        # if REF > 1, then the entire REF allele must be removed (across all positions) and replaced with the ALT allele
                        # do this with a dummy character, X, which will be later removed. 
                        # This is generalizable to even the case where REF == 1 because it will just replace the first index

                        # replace everything with X first
                        new_seq[idx:idx+len(ref_allele)] = "X" * len(ref_allele)

                        # then make the first position the alternative allele
                        new_seq[idx] = alt_allele
    
                    # don't do anything if indels are missing and very long
                    else:
                        print(fName, record.POS, record) # print for information purposes
                        continue

                # deletion -- include only if they are high quality. Leave low-quality deletions as reference
                else:

                    if single_allele_type == "alt":

                        # don't need to consider if the deletion only partially overlaps with the region of interest because already did that above.
                        # the replacement is the alternative allele padded with gap characters. # of gap characters = the length difference between them 
                        new_allele = '-' * (len(ref_allele) - len(alt_allele)) + alt_allele
                        assert len(new_allele) == len(ref_allele)
                        new_seq[idx:idx+len(ref_allele)] = new_allele

                        # # replace ref allele with X (will be removed later)
                        # old_len = len(new_seq)
                        # new_seq[idx:idx+len(ref_allele)] = 'X' * len(ref_allele)

                        # # then make the first position the alternative allele.
                        # new_seq[idx] = new_allele

                        # # # add this step so that if the allele extends more than the region of interest, it is truncated. This is for large deletions
                        # # new_seq = new_seq[:old_len]

    # sequence should be the same length because for both insertions and deletions, the new allele was inserted into a single list element, regardless of the length of the ref and alt alleles.
    if len(new_seq) != len(h37Rv_region):
        raise ValueError(fName, len(new_seq), len(h37Rv_region))
        exit()
        
    return new_seq


#################################### STEP 1: GET SNPS AND INDELS AND INSERT INTO EACH SEQUENCE USING THE FUNCTION ABOVE ####################################


print(f"Considering AFs > {AF_thresh} as present")

seq_dict = {}

for i, fName in enumerate(paths):
    
    seq_dict[os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", "")] = introduce_snps_indels_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=AF_thresh)

    # if len(paths) > 5000:
    #     if i % 1000 == 0:
    #         print(i)

#################################### STEP 2: FILL IN GAP CHARACTERS IN THE REFERENCE SEQUENCE ####################################

    
new_ref_seq = h37Rv_region.copy()

# get the reverse complement if negative sense. This function returns the joined sequence. If not, 
if SENSE == "NEG":
    new_ref_seq = reverse_complement(new_ref_seq)
else:
    new_ref_seq = "".join(new_ref_seq)
    
    
#################################### STEP 3: FILL IN GAP CHARACTERS IN ALL SEQUENCES AND WRITE TO THE OUTPUT FILE ####################################


if SAVE_FASTA:

    print(f"Writing aligned sequences to {OUT_FILE}")
    
    with open(OUT_FILE, "w+") as file:
    
        for isolate, seq in seq_dict.items():

            # sequence should be at least as long as the reference region. Can be longer due to structural variant insertions but not shorter because deletions are padded with '-'
            # assert len(seq) >= (END - START)
            if len(seq) < (END - START):
                os.remove(OUT_FILE)
                raise ValueError(isolate, len(seq), END - START)
                exit()
            
            # remove X characters, which are used for some insertions
            # check that the new length matches with the reference sequence, which has already had gap characters inserted
            seq = "".join(seq).replace("X", "")
        
            # get the reverse complement if negative sense
            if SENSE == "NEG":
                seq = reverse_complement(seq)
            
            # write the new sequence to the alignment file
            file.write(">" + isolate + "\n")
            file.write(seq + "\n")
        
        # write the reference sequence 
        file.write(">MT_H37Rv\n")
        file.write(new_ref_seq + "\n")

# read in and check if all sequences are identical
seq_df = [(seq.id, seq.seq) for seq in SeqIO.parse(OUT_FILE, "fasta")]
seq_df = pd.DataFrame(seq_df)
seq_df.columns = ['Isolate', 'Seq']

if seq_df['Seq'].nunique() == 1:
    print(f"All sequences in {OUT_FILE} are identical! Please exclude this locus from downstream models")

    # remove all files
    os.remove(OUT_FILE)
    os.remove(seq_dict_fName)
    os.remove(insertion_sites_fName)

# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")