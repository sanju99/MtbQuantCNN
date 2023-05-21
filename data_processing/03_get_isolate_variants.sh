#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-04:00
#SBATCH -p short
#SBATCH --mem=5G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

isolates_file=$1
isolates=($(cat "${isolates_file}"))
START=$2
END=$3
output_file=$4

# sbatch analysis/get_isolate_variants.sh MXF_isolates.txt 4998 9818 MXF_isolate_variants.tsv

# # create the output files
# echo -e "POS\tREF\tALT\tANN[0].GENE\tANN[0].HGVS_C\tANN[0].HGVS_P\tIsolate" > "$syn_output_file"
# echo -e "POS\tREF\tALT\tANN[0].GENE\tANN[0].HGVS_C\tANN[0].HGVS_P\tIsolate" > "$nonsyn_output_file"

# create the output files
echo -e "POS\tREF\tALT\tFILTER\tAF\tGENE\tEFFECT\tNUC\tPROT\tIsolate" > "$output_file"

# iterate through 
for isolate in "${isolates[@]}"; do

    # get all variants
    bcftools filter -i "POS >= $START & POS <= $END" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift extractFields --csv - POS REF ALT FILTER AF "ANN[0].GENE" "ANN[0].EFFECT" "ANN[0].HGVS_C" "ANN[0].HGVS_P" -s "," -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }' >> "$output_file"

#     # get synonymous variants
#     bcftools filter -i "POS >= $START & POS <= $END" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift filter "ANN[0].EFFECT = 'synonymous_variant'" | SnpSift extractFields - POS REF ALT "ANN[0].GENE" "ANN[0].HGVS_C" "ANN[0].HGVS_P"  -s "," -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }' >> "$syn_output_file"
    
#     # get all nonsynoymous variants
#     bcftools filter -i "POS >= $START & POS <= $END" "/n/scratch3/users/s/sak0914/annotated_VCF/$isolate.eff.vcf" | SnpSift filter "ANN[0].EFFECT != 'synonymous_variant'" | SnpSift extractFields - POS REF ALT "ANN[0].GENE" "ANN[0].HGVS_C" "ANN[0].HGVS_P"  -s "," -e "." | tail -n +2 | awk -v sample="$isolate" 'BEGIN { OFS="\t" } { print $0, sample }' >> "$nonsyn_output_file"
    
done