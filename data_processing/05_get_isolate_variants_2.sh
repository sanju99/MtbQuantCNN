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

output_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/$drug/isolate_variants.tsv"

# check if the output file exists. If not, create it.
if [ ! -f "$output_file" ]; then
    echo -ne "POS\tREF\tALT\tFILTER\tAF\tGENE\tEFFECT\tNUC\tPROT\tIsolate\n" > "$output_file"
else
    echo "Appending results to existing output file $output_file"
fi

# iterate through the file names and append the variants for each isolate to the output file. Whether or not the output file previously existed,
for fName in "${fNames[@]}"; do

    # extract isolate name from the full path to the full .vcf.gz file. Get the basename, then split on the "_" character and get the first bit
    # (because the file names are $isolate_full.vcf.gz)
    isolate=$(basename "$fName" | cut -d "_" -f 1)

    # Use awk to check if the isolate is present in the first column, which is Isolate
    found_isolate=$(awk -F'\t' -v search="$isolate" -v col="-1" '$NF == search' $output_file)

    # Check if the output is empty (string not found)
    if [ -z "$found_isolate" ]; then
        # Perform actions when the string is not found
        echo "Getting variants for $isolate"

        # INH regions: 4
        # bcftools filter -i "(POS > 2153234 & POS <= 2157381) | (POS > 1673299 & POS <= 1675011) | (POS > 2725477 & POS <= 2726780) | (POS > 2516548 & POS <= 2520712)" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields --csv - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -s "," -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }' >> "$output_file"

        # PZA regions: 4
        # isolate_variants=$(bcftools filter -i "(POS > 2287884 & POS <= 2291268) | (POS > 4043041 & POS <= 4046302) | (POS > 1833380 & POS <= 1836236) | (POS >= 4038051	& POS <= 4040980)" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }')

        # LEV and MXF
        # isolate_variants=$(bcftools filter -i "POS >= 4998 & POS <= 9818" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }')

        # # ETH: 3 loci
        # isolate_variants=$(bcftools filter -i "(POS >= 4326004 & POS <= 4327548) | (POS >= 4327474 & POS <= 4328199) | (POS >= 1673300 & POS <= 1675011)" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }')

        # KAN: 2 loci
        # isolate_variants=$(bcftools filter -i "(POS >= 2714124 & POS <= 2716394) | (POS >= 4327474 & POS <= 4328199)" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }')

        # LZD: 1 locus
        isolate_variants=$(bcftools filter -i "POS >= 800793 & POS <= 801462" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }')

        # If the output is NOT empty (i.e. there are variants), then append to the dataframe
        if [ ! -z "$isolate_variants" ]; then
            echo "$isolate_variants" >> "$output_file"
        # if it is empty, then append row of NaNs as a placeholder
        else
            # Create a row of NaN values
            nan_row=$(echo -e "NaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\tNaN\t$isolate")
            
            # Append the nan_row to the TSV file
            echo "$nan_row" >> "$output_file"
        fi
    
    fi
    
done

conda deactivate