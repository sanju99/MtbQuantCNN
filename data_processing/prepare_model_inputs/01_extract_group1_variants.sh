#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=5G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu


#################### REQUIRED COMMAND LINE ARGUMENTS (IN THIS ORDER): ####################


set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics # CHANGE TO YOUR OWN ENVIRONMENT OR REMOVE IF RUNNING IN THE BASE ENV


################################################################################################################################################

# if ! [ $# -eq 1 ]; then
#     echo "Please pass in 1 command line arguments: an input file of all absolute paths to VCF files"
#     exit
# fi

input_file="./samples_fNames_for_extraction.tsv"
output_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/group1_WHO_variants.tsv"

genes_file="./data_utils/Group1_WHO_V2_genes.txt"
noncoding_pos_file="./data_utils/Group1_WHO_V2_upstream_pos.txt"

# create output file with headers
# IMPORTANT: Need to put the newline character at the end so that the first new row is appended to the end, not concatenated as new columns!
if [ ! -f $output_file ]; then
    echo -ne "POS\tREF\tALT\tFILTER\tQUAL\tAF\tDP\tBQ\tMQ\tGENE_ID\tGENE\tEFFECT\tNUC\tPROT\tISOLATE\n" >> $output_file
fi

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r fName
do

    # get the sample name, basename then remove any file extensions
    basename_name=$(basename "$fName")

    # remove suffixes. Don't split on the _ character because that might be in some of the sample IDs
    sample_ID="${basename_name/_variants_combinedCodons.eff.vcf/}"

    # Use awk to check if the isolate is present in the last column, which is Isolate
    found_isolate=$(awk -F'\t' -v search="$sample_ID" -v col="-1" '$NF == search' $output_file)

    # Check if the output is empty (string not found)
    if [ -z "$found_isolate" ]; then

        # extract all candidate Group 1 variants, both in genes and promoter regions
        # this gets all variants in genes with ANY Group 1 variant and all non-coding sites with a Group 1 variant. So need to do further filtering on the mutations within genes to get the Group 1 variants.
        # -e flag is what should be listed for empty fields. In this case, we are using the empty string. So when you read it into pandas, these will be replace with NaN rather than a placeholder string
        # also need to add the sample ID to the dataframe, then add to the save file
        # only filter by and get the first annotation (not all the downstream effects) for each one
        isolate_variants=$(SnpSift filter --set $genes_file --set $noncoding_pos_file "((ANN[0].GENE in SET[0]) | (ANN[0].GENEID in SET[0]) | (POS in SET[1])) & FILTER=='PASS' & DP >= 5 & AF >= 0.75 & BQ >= 20 & QUAL >= 10" -f $fName | SnpSift extractFields '-'  POS REF ALT FILTER QUAL AF DP BQ MQ "ANN[0].GENEID" "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "" | tail -n +2 | awk -v sample="$sample_ID" 'BEGIN { OFS="\t" } { print $0, sample }')
    
        echo "$isolate_variants" >> "$output_file"

    else
        echo "Already extracted variants for $sample_ID"
        
    fi

done < "$input_file"