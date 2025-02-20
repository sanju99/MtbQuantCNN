#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-01:00
#SBATCH -p short
#SBATCH --mem=5G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics

# command line arguments
# input_file="./VCFs_remove_duplicate_indels.csv"
# out_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF_clean"

# input_file="/home/sak0914/MtbQuantCNN/VCFs_remove_duplicate_indels.csv"
# out_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"

# # Check if the output directory exists. If not, raise an error
# if [ ! -d "$out_dir" ]; then
#     echo "Output directory $out_dir doesn't exist!"
#     exit 1
# fi

# # Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
# while IFS=$',', read -r sample_ID fName exclude_pos
# do
#     # left-align indels, then get only variants, then finally remove the second variant (the structural variant) and write the output
#     bcftools norm --fasta-ref /home/sak0914/Mtb_Megapipe/references/ref_genome/H37Rv_NC_000962.3.fna "/n/data1/hms/dbmi/farhat/rollingDB/genomic_data/$sample_ID/pilon/${sample_ID}_full.vcf.gz" | bcftools view --types snps,indels,mnps,other | bcftools filter -i "$exclude_pos" > "$out_dir/$sample_ID.vcf"

#     # bcftools filter -i "$exclude_pos" $fName > "$out_dir/$sample_ID.vcf"
    
# done < "$input_file"


# input_file="/home/sak0914/MtbQuantCNN/VCFs_remove_duplicate_indels.csv"
input_file="/home/sak0914/MtbQuantCNN/all_samples_MIC_models.csv"
out_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF"

# Check if the output directory exists. If not, raise an error
if [ ! -d "$out_dir" ]; then
    echo "Output directory $out_dir doesn't exist!"
    exit 1
fi

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$',', read -r sample_ID variants_vcf full_vcf
do
    # chrom_name=$(zgrep -v '^#' $variants_vcf | awk 'NR==1 {print $1}')

    # if [ "$chrom_name" = "Chromosome" ]; then
    #     ref_genome="/home/sak0914/Mtb_Megapipe/references/ref_genome/H37Rv_NC_000962.3.fna"
    # else
    #     ref_genome="/n/data1/hms/dbmi/farhat/mtb_data/h37rv/h37rv.fna"
    # fi
    
    # bcftools norm --fasta-ref $ref_genome --rm-dup none $full_vcf | bcftools view --types snps,indels,mnps,other > $variants_vcf

    # echo "Finished left-aligning variants in $sample_ID with $ref_genome"

    bcftools sort "$variants_vcf" -o "$variants_vcf"
    echo "Finished sorting variants in $sample_ID"

    cp $variants_vcf "$out_dir/$sample_ID.vcf"
    
done < "$input_file"