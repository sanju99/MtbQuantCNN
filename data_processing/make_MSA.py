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
parser.add_argument('--save-debug-files', dest='SAVE_DEBUG_FILES', action='store_true', help="Flag to determine if the insertion sites CSV and full seq dict pickle file should be saved. These are for debugging, so don't use this flag if you're not debugging")
parser.add_argument('--f2', type=str, dest='ADDITIONAL_ISOLATES_FILE', help='Text file of additional VCFs to include in alignment')
parser.add_argument('--insilico-muts-file', dest='INSILICO_MUTS_FILE', type=str, help='Text file of insilico mutations to include in alignment')
parser.add_argument('--AF-thresh', type=float, dest='AF_THRESH', default=0.75, help='Alternative allele frequency threshold (exclusive) to consider variants present')

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
SAVE_DEBUG_FILES = cmd_line_args.SAVE_DEBUG_FILES
ADDITIONAL_ISOLATES_FILE = cmd_line_args.ADDITIONAL_ISOLATES_FILE
INSILICO_MUTS_FILE = cmd_line_args.INSILICO_MUTS_FILE
AF_THRESH = cmd_line_args.AF_THRESH # variants with an AF > AF_THRESH are considered present. AF ≤ 1 - AF_THRESH is absent. Everything else is missing

if START >= END:
    raise ValueError(f"START coordinate must be less than END coordinate. You passed in START = {START} and END = {END}")

SENSE = SENSE.upper()

if SENSE not in ["POS", "NEG"]:
    raise ValueError(f"SENSE argument must be one of pos, neg, POS, NEG. You passed in {SENSE}")

# must be a float less than or equal to 1
if AF_THRESH > 1:
    AF_THRESH /= 100
    
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
print(f"Making multiple sequence alignment for {len(paths)} sequences and {len(add_paths)} additional sequences for {os.path.basename(OUT_FILE).replace('.fasta', '')}")
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
print(f"Unaligned region size: {len(h37Rv_region)}")
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
        2. FILTER == Amb and 0.25 < AF <= 0.75
        3. SNP quality < 10

    Criteria for not confident in a variant or can not be reliably inserted, so leave it as reference:

        1. IMPRECISE variant (in the INFO field)
        2. Indels longer than 15 bp where neither the REF nor the ALT are of length 1 (this case is handled in the next function)
        
    If FILTER contains Amb and the alternative allele fraction > 0.9, then it is a pure alternative call. 
    If FILTER contains Amb and the alternative allele fraction < 0.1, the it is a pure reference call. 
    '''

    ref_allele = str(record.REF)
    alt_allele = "".join(np.array(record.ALT).astype(str))

    # this should not happen in pilon because it is not a haplotype variant caller
    # this would mean that there are 3 alleles present -- reference + 2 alternative alleles
    # haplotype variant callers will often have reference and alternatie haplotypes separated by a comma in the ALT field, so this script will not work for them
    if ',' in alt_allele:
        print(fName, record)
        raise ValueError(f"There are multiple alternative alleles in a single record!")
    
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
        
    # don't include IMPRECISE variants because they are difficult to reliably insert and often aren't reliable calls anyway
    # unreliability can be due to ambiguous alignments, complex genomic regions, low sequencing coverage, assembly gaps, or segmental duplications
    # basically these are breakpoints that the variant caller is not confident in. If we put Ns, often we get huge runs of Ns, which causes too much noise for the model.
    # pilon was not able to resolve the variants (usually due to large deletions), so leave as reference because we don't know what the variant is with high confidence

    # this occurs if there is a deletion upstream of this variant. This variant doesn't actually exist because there is no nucleotide there
    # it has been deleted by a deletion upstream. So don't 
    if "Del" in record.FILTER and len(ref_allele) == len(alt_allele):
        return "ref"
        
    if "IMPRECISE" in record.INFO.keys():
        return "ref"
    
    # the filter field is an empty list of it is PASS, else the list is non-empty
    # only consider the non-Amb cases here. Amb cases will be later, check the AF too for that
    if len(record.FILTER) > 0 and "Amb" not in record.FILTER:
        return "missing"
    
    # because IMPRECISE is taken care of above, this should only return missing for cases where REF = N or ALT = N
    if "N" in ref_allele or "N" in alt_allele:
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
    if len(ref_allele) == len(alt_allele) and 'BQ' in record.INFO.keys():
        if record.INFO['BQ'] < 20:
            return 'missing'

    # at this point, we have already checked all of the low-quality criteria. If a variant has made it this far without returning anything, then it is high quality
    # except for the AF, which will be handled differently for inframe indels, frameshift indels, and SNVs/MNVs later
    # PASS or Amb filters and an alternative allele fraction of less than 0.75 means we have a mixture of REF and ALT
    if "AF" in record.INFO.keys():
        # ≤ 25%, always absent
        if af <= 0.25:
            return "ref"
        # the other 2 depend on the threshold passed in
        elif af > presentThresh:
            return "alt"
        else:
            # check that the number of reads supporting the insertion or deletion is at least 5 then. 
            # structural variants may not have IC or DC in the INFO field, but they also don't have AF in the INFO field, so they wouldn't get to this loop anyway

             # for indels, encode them as present if they fail only the AF threshold because sometimes that happens because of read clipping. The indel is present, but if many of the reads are near the edge here, then they won't have the indel in them, and the AF is artificially low.
            # but double check that at least 5 reads support the indel
            if len(alt_allele) > len(ref_allele):
                if record.INFO['IC'] >= 5:        
                    return "lowAF"
                    
            elif len(ref_allele) > len(alt_allele):
                if record.INFO['DC'] >= 5:        
                    return "lowAF"

            # low AF SNVs/MNVs remain as missing
            else:
                return "missing"
    
    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF <= 0.25
    return "alt"



def introduce_snps_indels_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=0.75):
    
    new_seq = h37Rv_region.copy()

    # these VCF files contain very few variants, so no need to run tabix and subset the VCF file
    if 'inSilico_analysis' in fName or 'synthetic_VCF' in fName:

        # get all records in the file. There should only be 1 because MNPs were combined into a single variant to get correct snpEff annotations
        vcf_reader = vcf.Reader(filename=fName)
        records = [record for record in vcf_reader]
        assert len(records) == 1
    
    else:
    
        # create the tabix file if it doesn't exist
        if not os.path.isfile(f"{fName}.bgz.tbi"):
    
            print(f"Creating tabix file for {fName}")
    
            # bgzip the VCF file. NEED TO PUT "" AROUND FILE NAME TO PROPERLY CONSIDER SPECIAL CHARACTERS IN FILENAME
            if not os.path.isfile(f"{fName}.bgz"):
                subprocess.run(f'bgzip -c "{fName}" > "{fName}".bgz', shell=True)
    
            # tabix the bgzipped file, which will create fName.bgz.tbi
            subprocess.run(f'tabix -0 -p vcf "{fName}".bgz -f', shell=True)
    
        # VCF file was indexed using 0-indexed half-open scheme, so keep START and END coords as they are
        vcf_reader = vcf.Reader(filename=f"{fName}.bgz", compressed=True)
    
        # need to read in the bgzipped file in order to use fetch
        try:
            records = vcf_reader.fetch('NC_000962.3', start=START, end=END)
        except:
            records = vcf_reader.fetch('Chromosome', start=START, end=END)

    # start is 0-indexed (exclusive) and end is 1-indexed (inclusive)
    for record in records:

        # convert alternative allele from list to string
        alt_allele = "".join(np.array(record.ALT).astype(str))
        ref_allele = str(record.REF)
        variant_start = record.POS # 1-index space

        length_diff = np.abs(len(ref_allele) - len(alt_allele))

        # the index to replace -- this is 0-indexed, consistent with Python
        idx = variant_start - (START + 1)
        
        # get the allele type: ref, alt, or missing
        single_allele_type = allele_category(record, qualThresh, presentThresh)
        
        # frameshift indels can not be assigned a "missing" tag. They must be either ref or alt because there are too many potential problems
        # two such problems: 1) if you encode "missing" indels with Ns, then when they are translated to AA space, the Ns get translated. This can be wrong, especially on the last nucleotide of a codon, i.e. CCN = Proline due to the wobble effect. This will be mis-encoded
        
        # We would need to introduce another character for missing indel. But then when you translate the sequence, what then? Do you translate the deletion or the reference? That will considerably change the AA sequence. If you replace the new character with '', then you are treating the deletion present. 
        
        # to get around it, you may have to insert multiple sequences for the same isolate, weighted by the support
        
        # in preprocessing, we already removed isolates containing multiple ambiguous frameshifts in a single gene because this could be due to distinct clones (of the same lineage) in an isolate. Because < 75% of reads support both frameshifts, they both may not be present in a single sample, so the sequence will be incorrect.

        if length_diff != 0:

            # frameshifts
            if length_diff % 3 != 0:

                # low-QC inframe indels can be left as missing, but low-QC frameshifts must be reverted to reference because there's no good way to encode them when missing
                if single_allele_type == 'missing': 
                    
                    print(os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", ""), record, 'REF!')
                    single_allele_type = 'ref'
    
            # encode these as present variants because they pass the other QC thresholds, but may have low AF due to an alignment artifact
            # this is the same for both inframe and frameshift indels
            if single_allele_type == 'lowAF':
                single_allele_type = 'alt'
                print(os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", ""), record, 'ALT!')

        # no more lowAF should remain
        assert single_allele_type in ['ref', 'missing', 'alt']

        # only change the sequence if the type is not reference
        if single_allele_type != "ref":

            # variant occurs upstream of the region of interest but it still affects the region of interest
            # then need to remove that many nucleotides from the REF and ALT.
            # should work for all cases because if len(ALT) or len(REF) < abs(idx), then the allele will be the empty string
            # the length of that is 0, so it should work fine with insertions and deletions. 
            if idx < 0:

                # the starting nucleotide up to the START - 1, inclusive, nucleotides need to be removed
                num_nt_remove_beginning = START - variant_start + 1
                
                # remove the first N nucleotides from REF and ALT if that's the case
                ref_allele = ref_allele[num_nt_remove_beginning:]
                alt_allele = alt_allele[num_nt_remove_beginning:]
                    
                # then make the index 0 because now we've shifted the record to the start of the region of interest
                # also adjust the start from record.POS to START
                idx = 0
                variant_start = START + 1 # add 1 because variant_start is in 1-index space
            
            # if the variant extends past the region of interest, then need to truncate to avoid length mismatches
            # need to add 1 because the ends are exclusive when indexing the strings/lists
            # the variant must extend BEYOND END. If it is exactly at END, then no adjustment is needed. That's why the logic is > not ≥
            if variant_start + len(ref_allele) > END:# + 1:

                extra_len_ref = variant_start + len(ref_allele) - END

                # add 1 because the end is exclusive so need to extend the alleles by one more position
                ref_allele = ref_allele[:-extra_len_ref + 1]

            if variant_start + len(alt_allele) > END:# + 1:
                
                extra_len_alt = variant_start + len(alt_allele) - END

                # add 1 because the end is exclusive so need to extend the alleles by one more position
                alt_allele = alt_allele[:-extra_len_alt + 1]

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
                
                # insertions
                if len(alt_allele) > len(ref_allele):
                    
                    # replace the nucleotide at the reference index with the alternative nucleotides
                    # also add the number of gap characters needed (len(ALT) - len(REF) to insertion_dict at the appropriate index                        
                    # only input insertions up to 100 bp and also if they pass the QC filters. Leave the others as reference to avoid introducing long runs of Ns into the aln
                    # if single_allele_type == "alt" or (single_allele_type == "missing": # and (len(alt_allele) - len(ref_allele) <= 100)):
                            
                    if single_allele_type == "missing":
                        alt_allele = "N"*len(alt_allele)

                    # if REF > 1, then the entire REF allele must be removed (across all positions) and replaced with the ALT allele
                    # do this with a dummy character, X, which will be later removed. 
                    # This is generalizable to even the case where REF == 1 because it will just replace the first index
                    # replace everything with X first
                    new_seq[idx:idx+len(ref_allele)] = "X" * len(ref_allele)

                    # then make the first position the alternative allele
                    new_seq[idx] = alt_allele

                    insertion_dict[idx] = np.max([insertion_dict[idx], len(alt_allele) - len(ref_allele)])

                # deletions
                else:

                    # if single_allele_type == "alt" or (single_allele_type == "missing": # and (len(ref_allele) - len(alt_allele) <= 100)):

                    # don't need to consider if the deletion only partially overlaps with the region of interest because already did that above.
                    # the replacement is the alternative allele padded with gap characters. # of gap characters = the length difference between them 
                    # leave the remaining allele (alt) as itself, and make the deletion N
                    if single_allele_type == "missing":
                        new_allele = alt_allele + 'N' * (len(ref_allele) - len(alt_allele))

                    else:
                        new_allele = alt_allele + '-' * (len(ref_allele) - len(alt_allele))
                        
                    assert len(new_allele) == len(ref_allele)
                    
                    new_seq[idx:idx+len(ref_allele)] = new_allele

    # sequence should be the same length because for both insertions and deletions, the new allele was inserted into a single list element, regardless of the length of the ref and alt alleles.
    if len(new_seq) != len(h37Rv_region):
        raise ValueError(fName, len(new_seq), len(h37Rv_region))
        exit()
        
    return new_seq



#################################### STEP 1: GET SNPS AND INDELS AND INSERT INTO EACH SEQUENCE USING THE FUNCTION ABOVE ####################################


# keep track of positions and the numbers of insertions relative to H37Rv
# after this main loop, these need to be introduced into h37Rv_region and also into 
global insertion_dict
insertion_dict = dict(zip(np.arange(0, END-START), np.zeros(END-START)))
print(f"Considering AFs > {AF_THRESH} as present")

seq_dict = {}

for i, fName in enumerate(paths):
    
    seq_dict[os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", "")] = introduce_snps_indels_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=AF_THRESH)

    # if len(paths) > 5000:
    #     if i % 1000 == 0:
    #         print(i)

print(f"Finished reading {len(seq_dict)} sequences!")

# convert to dataframe for easy querying. Convert everything to integers and keep only indices where gap characters need to be inserted (num_insertion > 0)
insertion_sites = pd.DataFrame(insertion_dict, index=[0]).T.reset_index().rename(columns={"index": "aln_idx", 0:"len_insertion"})
insertion_sites = insertion_sites.query("len_insertion > 0").reset_index(drop=True)
insertion_sites[insertion_sites.columns] = insertion_sites[insertion_sites.columns].astype(int)

if SAVE_DEBUG_FILES:
    seq_dict_fName = os.path.join(os.path.dirname(OUT_FILE), f"{os.path.basename(OUT_FILE).split('.')[0]}_seq_dict.pkl")
    pd.DataFrame(seq_dict).to_pickle(seq_dict_fName)

# need this later for insilico mutagenesis, so always save
insertion_sites_fName = os.path.join(os.path.dirname(OUT_FILE), f"{os.path.basename(OUT_FILE).split('.')[0]}_insertion_sites.csv")
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


# remove training data isolates
training_isolates = [os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", "") for fName in pd.read_csv(PATHS_FILE, sep="\t", header=None)[0].values]

if SAVE_FASTA:

    print(f"Writing aligned sequences to {OUT_FILE}")
    
    with open(OUT_FILE, "w+") as file:
    
        for isolate, seq in seq_dict.items():
    
            # length should match because we already inserted gap characters into H37Rv in the loop above and gap characters were already inserted into each sample sequence in the function
            # assert len(seq) >= (END - START)
            if len(seq) < (END - START):
                os.remove(OUT_FILE)
                raise ValueError(isolate, len(seq), END - START)
                exit()
                
            for _, row in insertion_sites.iterrows():
    
                # the numbers in insertion_sites are the number of nucleotides to insert, i.e. len(alt) - len(ref)
                # for nearly all insertions, the length of the REF allele is 1 so add 1 to this.
                site_length = row["len_insertion"] + 1
    
                # if the length of the site in a given isolate is smaller than the number of inserted nucleotides, then pad the end with gap characters
                if len(seq[row["aln_idx"]]) < site_length:
                    seq[row["aln_idx"]] = seq[row["aln_idx"]] + "-" * int(site_length - len(seq[row["aln_idx"]]))
    
                # these are the few cases where len(REF) > 1, so they would fail a check here. But the final sequence length will be checked against the reference below
                elif len(seq[row["aln_idx"]]) > site_length:
                    continue
                    
            # remove X characters, which are used for some insertions
            # check that the new length matches with the reference sequence, which has already had gap characters inserted
            seq = "".join(seq).replace("X", "")
            if len(seq) != len(new_ref_seq):
                print(seq)
                os.remove(OUT_FILE)
                raise ValueError(isolate, len(seq), len(new_ref_seq))
                exit()
        
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
seq_df = pd.DataFrame([(seq.id, seq.seq) for seq in SeqIO.parse(OUT_FILE, "fasta")])
seq_df.columns = ['Isolate', 'Seq']

if seq_df['Seq'].nunique() == 1 and ADDITIONAL_ISOLATES_FILE is None and INSILICO_MUTS_FILE is None:
    
    print(f"All sequences in {OUT_FILE} are identical! Please exclude this locus from downstream models")

    # remove all files
    os.remove(OUT_FILE)
    os.remove(insertion_sites_fName)

    if os.path.isfile(seq_dict_fName):
        os.remove(seq_dict_fName)
    
# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")