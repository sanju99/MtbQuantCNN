#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-01:00
#SBATCH -p short
#SBATCH --mem=20G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate bioinformatics

# training samples only
vcf_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"
isolates_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/fNames.txt"

# ls $vcf_dir > $isolates_file

fNames=($(cat "${isolates_file}"))
echo "Getting variants for ${#fNames[@]} isolates"
output_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/trainVal_isolateVariants_AllDrugs.tsv"

# check if the output file exists. If not, create it.
if [ ! -f "$output_file" ]; then
    echo -ne "POS\tREF\tALT\tFILTER\tAF\tGENE\tEFFECT\tNUC\tPROT\tIsolate\n" > "$output_file"
else
    echo "Appending results to existing output file $output_file"
fi

# iterate through the file names and append the variants for each isolate to the output file. Whether or not the output file previously existed,
for fName in "${fNames[@]}"; do

    # extract isolate name from the full path to the full .vcf file. Get the basename, then split on the "." character and get the first part to remove the file extension
    isolate=$(basename "$fName" | cut -d "." -f 1)

    # Use awk to check if the isolate is present in the last column, which is Isolate
    found_isolate=$(awk -F'\t' -v search="$isolate" -v col="-1" '$NF == search' $output_file)

    # Check if the output is empty (string not found)
    if [ -z "$found_isolate" ]; then

        echo "Getting variants for $isolate"
        
        isolate_variants=$(SnpSift extractFields "$vcf_dir/$isolate/pilon/$isolate.eff.vcf" POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }')
        
        echo "$isolate_variants" >> "$output_file"

    else

        echo "Already got variants for $isolate"
    
    fi
    
done