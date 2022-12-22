import sys, glob, os, vcf, tracemalloc
import numpy as np
import pandas as pd
from Bio import SeqIO

# starting the memory monitoring
tracemalloc.start()


###### STEP 0: READ IN FILES AND INITIALIZE VARIABLES ######
    
    
_, PHENOS_FILE, VCF_DIR, START, END, SENSE, OUT_FILE = sys.argv

START = int(START)
END = int(END)

SENSE = SENSE.upper()
assert SENSE in ["POS", "NEG"]

if not os.path.isfile(PHENOS_FILE):
    raise ValueError(f"{PHENOS_FILE} is not a file!")
    
isolates = pd.read_csv(PHENOS_FILE)["ROLLINGDB_ID"].values
paths = [os.path.join(VCF_DIR, isolate) + ".vcf" for isolate in isolates]
print(f"Making multiple sequence alignment for {len(paths)} sequences")
  
if ".fasta" not in OUT_FILE:
    OUT_FILE = OUT_FILE.split(".")[0] + ".fasta"
    
# H37Rv reference strain
h37Rv = SeqIO.read("/n/data1/hms/dbmi/farhat/Sanjana/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")
genome_len = len(h37Rv)
print(f"Reference genome size: {genome_len}")

# get only the region of interest
h37Rv_region = list(str(h37Rv.seq[START:END]))
print(f"Region size: {len(h37Rv_region)}")
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
    seq = [comp_dict[base] for base in seq] 
    
    # reverse the sequence and return as a list
    return "".join(seq[::-1])

    
    
def allele_category(record, qualThresh=10, heteroThresh=0.25):
    '''
    Returns "alt" or "ref" if the variant is low-quality or ambiguous. Otherwise this function returns "missing"
    
    Low-quality criteria:
    
        1. FILTER == Del, LowCov
        2. FILTER == Amb and 0.1 < AF < 0.9
        3. IMPRECISE variant (in the INFO field)
        4. SNP quality < 10
        
        
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
        
    if record.QUAL is None:
        qual = 11
    else:
        qual = record.QUAL
        
    if "IMPRECISE" in record.INFO.keys():
        return "missing"
    
    # the filter field is an empty list of it is PASS, else the list is non-empty
    # only consider the non-Amb cases here. Amb cases will be later, check the AF too for that
    if len(record.FILTER) > 0 and "Amb" not in record.FILTER:
        return "missing"
    
    # ambiguous reference or alternative alleles. I think these are very rare though
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
        
    # low SNP quality or multiple bad FILTER tags. i.e. if the field is Del;Amb, it will be ['Del', 'Amb'] in pyVCF. The length of this will be >= 2
    if qual < qualThresh or len(record.FILTER) > 1:
        return "missing"
        
    # if nothing has been returned, then the variant is high quality (there are no REF = ALT records in the input VCF files, so return the alternative variant)
    # the reference variant only gets returned above if FILTER == Amb and AF < 0.25
    return "alt"




def introduce_snps_indels_single_seq(fName, h37Rv_region, START, END):
    
    new_seq = h37Rv_region.copy()

    vcf_file = vcf.Reader(filename=fName)

    # start is 0-indexed (exclusive) and end is 1-indexed (inclusive)
    for record in vcf_file:

        # get the allele type: ref, alt, or missing
        single_allele_type = allele_category(record) 

        # get only the region of interest
        if record.POS > START and record.POS <= END:

            # convert alternative allele from list to string
            alt_allele = "".join(np.array(record.ALT).astype(str))

            # the index to replace -- this is 0-indexed, consistent with Python
            idx = record.POS - (START + 1)

            # no length change -- SNP or MNP. Python will replace all elements if the original and new are the same length
            if len(record.REF) == len(alt_allele):

                if single_allele_type == "alt":
                    new_seq[idx:idx+len(record.REF)] = alt_allele
                elif single_allele_type == "missing":
                    new_seq[idx:idx+len(record.REF)] = ["N"]*len(alt_allele)
                # the only other option is reference, so don't do anything
                else:
                    continue

            # indels
            else:

                # insertion
                if len(alt_allele) > len(record.REF):

                    if len(record.REF) == 1:

                        # simply replace the nucleotide at the reference index with the alternative nucleotides
                        # also add the number of gap characters needed (len(ALT) - len(REF), where len(REF) == 1) to insertion_dict at the appropriate index
                        # only consider high-quality insertions and deletions. Leave the others as reference
                        if single_allele_type == "alt":

                            new_seq[idx] = alt_allele
                            insertion_dict[idx] = np.max([insertion_dict[idx], (len(alt_allele) - 1)])
                        
                        # don't do anything if reference or missing. Don't consider missing indels because they often introduce huge regions of N
                        else:
                            continue

                    # complex variant -- don't process these
                    else:
                        continue

                # deletion
                else:

                    if len(alt_allele) == 1:

                        ref_allele = str(record.REF)
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
                        if single_allele_type == "alt":
                            new_seq[idx:idx+len(ref_allele)] = new_allele
                        
                        # don't do anything if reference or missing. Don't consider missing indels because they often introduce huge regions of N
                        else:
                            continue

                    # complex variant -- don't process these
                    else:
                        continue

    # check lengths because both of them are lists right now 
    assert len(new_seq) == len(h37Rv_region)
    
    return new_seq



###### STEP 1: GET SNPS AND INDELS AND INSERT INTO EACH SEQUENCE USING THE FUNCTION ABOVE ######


# keep track of positions and the numbers of insertions relative to H37Rv
# after this main loop, these need to be introduced into h37Rv_region and also into 
global insertion_dict
insertion_dict = dict(zip(np.arange(0, END-START), np.zeros(END-START)))

seq_dict = {}

for i, fName in enumerate(paths):
    
    seq_dict[os.path.basename(fName).split(".")[0]] = introduce_snps_indels_single_seq(fName, h37Rv_region, START, END)

    if i % 1000 == 0:
        print(i)

print(f"Finished reading {len(seq_dict)} sequences!")

# convert to dataframe for easy querying. Convert everything to integers and keep only indices where gap characters need to be inserted (num_insertion > 0)
insertion_sites = pd.DataFrame(insertion_dict, index=[0]).T.reset_index().rename(columns={"index": "idx", 0:"len_insertion"})
insertion_sites = insertion_sites.query("len_insertion > 0").reset_index(drop=True)
insertion_sites[insertion_sites.columns] = insertion_sites[insertion_sites.columns].astype(int)
print(insertion_sites)


###### STEP 2: FILL IN GAP CHARACTERS IN THE REFERENCE SEQUENCE ######


new_ref_seq = h37Rv_region.copy()

for _, row in insertion_sites.iterrows():
    
    # number of gap characters to add
    add_gap = row["len_insertion"]

    new_ref_seq[row["idx"]] = new_ref_seq[row["idx"]] + "-" * add_gap
        
print(len(new_ref_seq), len("".join(new_ref_seq)))

# get the reverse complement if negative sense
if SENSE == "NEG":
    new_ref_seq = reverse_complement(new_ref_seq)


###### STEP 3: FILL IN GAP CHARACTERS IN ALL SEQUENCES AND WRITE TO THE OUTPUT FILE ######


with open(OUT_FILE, "w+") as file:

    for isolate, seq in seq_dict.items():

        assert len(seq) == (END - START)

        for _, row in insertion_sites.iterrows():

            # the numbers in insertion_sites have 1 subtracted them, so they are the number of nucleotides to insert
            # this is the length that that position should be
            pos_length = row["len_insertion"] + 1

            # compare the lengths of the nucleotides at the given index and the pos_length (with gap characters)
            if len(seq[row["idx"]]) < pos_length:

                seq[row["idx"]] = seq[row["idx"]] + "-" * int(pos_length - len(seq[row["idx"]]))

            assert len(seq[row["idx"]]) == pos_length

        # check that the new length matches with the reference sequence, which has already had gap characters inserted
        assert len("".join(seq)) == len(new_ref_seq)

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