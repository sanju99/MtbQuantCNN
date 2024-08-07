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

if ! [ $# -eq 1 ]; then
    echo "Please pass in 1 command line argument: the drug abbreviation"
    exit
fi

drug="$1"

data_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs"

input_file="$data_dir/$drug/test_set_isolates.txt"
output_file="$data_dir/$drug/group_1_2_candidate_variants.tsv"

genes_file="$data_dir/$drug/group_1_2_genes.txt"
noncoding_pos_file="$data_dir/$drug/group_1_2_noncoding_pos.txt"

if [ ! -f $input_file ]; then
    echo "Input file $input_file does not exist!"
    exit
fi

if [ ! -f $genes_file ]; then
    echo "Genes file $genes_file does not exist!"
    exit
fi

if [ ! -f $noncoding_pos_file ]; then
    echo "Noncoding positions file $noncoding_pos_file does not exist!"
    exit
fi

# remove output file if it already exists
if [ -f $output_file ]; then
    rm $output_file
fi

# create output file with headers
# IMPORTANT: Need to put the newline character at the end so that the first new row is appended to the end, not concatenated as new columns!
echo -ne "POS\tREF\tALT\tFILTER\tQUAL\tAF\tDP\tBQ\tMQ\tGENE_ID\tGENE\tEFFECT\tNUC\tPROT\tISOLATE\n" >> $output_file

echo "Getting variants for test set isolates in $input_file and saving to $output_file"

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID
do
    
    fName="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF_clean_annot/${sample_ID}_combinedCodons.eff.vcf"

    if [ ! -f $fName ]; then
        echo "Annotated VCF file $fName does not exist!"
        exit
    fi

    # sometimes empty if there are no Group 1/2 upstream noncoding mutations
    if [ ! -s $noncoding_pos_file ]; then

        isolate_variants=$(SnpSift filter --set $genes_file "((ANN[0].GENE in SET[0]) | (ANN[0].GENEID in SET[0])) & FILTER=='PASS' & QUAL >= 10 & DP >= 5 & AF >= 0.75 & BQ >= 20 & MQ >= 30" -f $fName | SnpSift extractFields '-'  POS REF ALT FILTER QUAL AF DP BQ MQ "ANN[0].GENEID" "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "" | tail -n +2 | awk -v sample="$sample_ID" 'BEGIN { OFS="\t" } { print $0, sample }')

    else

        isolate_variants=$(SnpSift filter --set $genes_file --set $noncoding_pos_file "((ANN[0].GENE in SET[0]) | (ANN[0].GENEID in SET[0]) | (POS in SET[1])) & FILTER=='PASS' & QUAL >= 10 & DP >= 5 & AF >= 0.75 & BQ >= 20 & MQ >= 30" -f $fName | SnpSift extractFields '-'  POS REF ALT FILTER QUAL AF DP BQ MQ "ANN[0].GENEID" "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "" | tail -n +2 | awk -v sample="$sample_ID" 'BEGIN { OFS="\t" } { print $0, sample }')

    fi
    
    echo "$isolate_variants" >> "$output_file"

done < "$input_file"