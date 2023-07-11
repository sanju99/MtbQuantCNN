#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short
#SBATCH --mem=5G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate bioinformatics

drug=$1
paths_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/$drug/paths.txt"
#val_paths_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/$drug/validation_paths.txt"

fNames=($(cat "${paths_file}"))
#val_fNames=($(cat "${val_paths_file}"))

# combine into a single array
#fNames+=(${val_fNames[@]})
echo "Getting $drug-relevant variants for ${#fNames[@]} isolates"

output_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/$drug/isolate_variants_FULL.tsv"

# check if the output file exists. If not, create it.
if [ ! -f "$output_file" ]; then
    echo -ne "POS\tREF\tALT\tFILTER\tQUAL\tAF\tGENE\tEFFECT\tNUC\tPROT\tIsolate\n" > "$output_file"
else
    echo "Appending results to existing output file $output_file"
fi

# iterate through the file names and append the variants for each isolate to the output file. Whether or not the output file previously existed,
for fName in "${fNames[@]}"; do

    # extract isolate name from the full path to the full .vcf.gz file. Get the basename, then split on the "_" character and get the first bit
    # (because the file names are $isolate_full.vcf.gz)
    isolate=$(basename "$fName" | cut -d "_" -f 1)

    SnpSift extractFields "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" POS REF ALT FILTER QUAL AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }' >> "$output_file"
    
done

conda deactivate