import argparse, subprocess, glob, os, vcf, tracemalloc
import numpy as np
import pandas as pd
from Bio import SeqIO, Seq

# starting the memory monitoring
tracemalloc.start()


#################################### STEP 0: READ IN FILES AND INITIALIZE VARIABLES ####################################
    
    
######## IMPORTANT: START is 0-indexed, END is 1-indexed (i.e., 0-indexed half-open) to be consistent with other bioinformatics tools ########
        

parser = argparse.ArgumentParser()

# Add a required string argument for the paths file
parser.add_argument("-i", type=str, dest='PATHS_FILE', help='Text file of VCF paths to include in alignment', required=True)

# dest indicates the name that each argument is stored in so that you can access it after running .parse_args()
parser.add_argument('-start', type=int, dest='START', help='Start coordinate for alignment (0-indexed, inclusive)', required=True)
parser.add_argument('-end', type=int, dest='END', help='End coordinate for alignment (1-indexed, exlusive)', required=True)
parser.add_argument('-sense', type=str, dest='SENSE', help='Sense, must be one of pos, neg, POS, NEG', required=True)
parser.add_argument('-o', type=str, dest='OUT_FILE', help='Name of the output FASTA file', required=True)

# optional arguments
parser.add_argument('--g', type=str, dest="GENOME_FILE", default="/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", help="Full path to Genbank file for the reference genome")
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

file_paths = pd.read_csv(PATHS_FILE, sep="\t", header=None)[0].values

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
print(f"Concatenating SNPs in a region of length {len(h37Rv_region)}")
del h37Rv    



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
        
    If FILTER contains Amb and the alternative allele fraction > presentThresh, then it is a pure alternative call. 
    '''

    ref_allele = str(record.REF)
    alt_allele = "".join(np.array(record.ALT).astype(str))

    # this should not happen in pilon because it is not a haplotype variant caller
    # this would mean that there are 3 alleles present -- reference + 2 alternative alleles
    # haplotype variant callers will often have reference and alternative haplotypes separated by a comma in the ALT field, so this script will not work for them
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
        
    # this occurs if there is a deletion upstream of this variant. This variant doesn't actually exist because there is no nucleotide there
    if "Del" in record.FILTER and len(ref_allele) == len(alt_allele):
        return "del"

    # don't include IMPRECISE variants because they are difficult to reliably insert and often aren't reliable calls anyway
    # unreliability can be due to ambiguous alignments, complex genomic regions, low sequencing coverage, assembly gaps, or segmental duplications
    # basically these are breakpoints that the variant caller is not confident in. If we put Ns, often we get huge runs of Ns, which causes too much noise for the model.
    # pilon was not able to resolve the variants (usually due to large deletions), so leave as reference because we don't know what the variant is with high confidence
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
    # ≤ 25%, always absent
    if af <= 0.25:
        return "ref"
    # present depends on the threshold passed in
    elif af > presentThresh:
        return "alt"

    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF <= 0.25
    return "alt"



def introduce_snps_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=0.75):
    
    new_seq = h37Rv_region.copy()

    # don't need to make tabix files if running whole-genome alignments
    if len(h37Rv_region) != genome_len:
        
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
    
        # need to read in the bgzipped file in order to use fetch. There's some variation in CHROM naming, so try both possibilities
        try:
            records = vcf_reader.fetch('NC_000962.3', start=START, end=END)
        except:
            records = vcf_reader.fetch('Chromosome', start=START, end=END)

    # read in all records
    else:
        records = vcf.Reader(filename=fName)

    # start is 0-indexed (exclusive) and end is 1-indexed (inclusive)
    for record in records:

        # only process SNPs with this script
        if check_is_snp(record):

            # convert alternative allele from list to string
            alt_allele = "".join(np.array(record.ALT).astype(str))
            ref_allele = str(record.REF)
            variant_start = record.POS # 1-index space
        
            # the index to replace -- this is 0-indexed, consistent with Python
            idx = variant_start - (START + 1)
            
            # get the allele type: ref, alt, or missing
            single_allele_type = allele_category(record, qualThresh, presentThresh)
            assert single_allele_type in ['ref', 'missing', 'alt', 'del']
    
            # only change the sequence if the type is not reference
            if single_allele_type != "ref":
        
                # no length change -- SNP or MNP. Python will replace all elements if the original and new are the same length
                if len(ref_allele) == len(alt_allele):
    
                    old_len = len(new_seq)
    
                    if single_allele_type == "alt":
                        new_seq[idx:idx+len(ref_allele)] = alt_allele
                    
                    elif single_allele_type == "missing":
                        new_seq[idx:idx+len(ref_allele)] = "N" * len(alt_allele)

                    elif single_allele_type == "del":
                        new_seq[idx:idx+len(ref_allele)] = "-" * len(alt_allele)
                    
                    # the only other option is reference, so don't do anything
                    else:
                        continue

        
    return new_seq



#################################### STEP 1: REVERSE COMPLEMENT THE REFERENCE SEQUENCE IF NEGATIVE SENSE ####################################

    
new_ref_seq = "".join(h37Rv_region.copy())

# get the reverse complement if negative sense.
if SENSE == "NEG":
    new_ref_seq = str(Seq.Seq(new_ref_seq).reverse_complement())


#################################### STEP 2: GET SNPS AND INSERT INTO EACH SEQUENCE USING THE FUNCTION ABOVE ####################################


# storing all sequences in memory is very memory-intensive if you're getting SNPs on the whole genome, so write the sequence to the FASTA file, then delete it
print(f"Considering AFs > {AF_THRESH} as present and writing {len(file_paths)} SNP concatenates to {OUT_FILE}")

with open(OUT_FILE, "w+") as file:

    for i, fName in enumerate(file_paths):

        # remove file extensions and other suffixes from the sequence name
        isolate_name = os.path.basename(fName).replace(".eff", "").replace(".vcf", "").replace("_variants", "")
        
        # remove extensions from the filename when adding the sample name to the sequences dictionary
        seq_with_snps = introduce_snps_single_seq(fName, h37Rv_region, START, END, qualThresh=10, presentThresh=AF_THRESH)

        # join the list into a string
        seq_with_snps = "".join(seq_with_snps)
        
        # get the reverse complement if negative sense
        if SENSE == "NEG":
            seq_with_snps = str(Seq.Seq(seq_with_snps).reverse_complement())

        # write the new sequence to the alignment file
        file.write(">" + isolate_name + "\n")
        file.write(seq_with_snps + "\n")

        # delete the sequence
        del seq_with_snps

        # if i % 100 == 0:
        #     print(i)

    # write the reference sequence as the last sequence
    file.write(">MT_H37Rv\n")
    file.write(new_ref_seq + "\n")
        
    
# returns a tuple: current, peak memory in bytes 
script_memory = tracemalloc.get_traced_memory()[1] / 1e9
tracemalloc.stop()
print(f"{script_memory} GB\n")