#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
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
input_file="/home/sak0914/MtbQuantCNN/analysis/L3_MXF_samples.csv"
out_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/L3_MXF"

# Check if the output directory exists. If not, raise an error
if [ ! -d "$out_dir" ]; then
    echo "Output directory $out_dir doesn't exist!"
    exit 1
fi

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$',', read -r sample_ID
do
# for vcf_fName in /n/data1/hms/dbmi/farhat/rollingDB/genomic_data/*/pilon/*_full_vcf.gz; do

    if [ ! -f $out_dir/$sample_ID.vcf ]; then
    
        chrom_name=$(zgrep -v '^#' "/n/data1/hms/dbmi/farhat/rollingDB/genomic_data/$sample_ID/pilon/${sample_ID}_full.vcf.gz" | awk 'NR==1 {print $1}')
    
        if [ "$chrom_name" = "Chromosome" ]; then
            bed_file="/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/RLC_Regions.Plus.LowPmapK50E4.H37Rv.ExtendSingleNucs.Chromosome.bed"
        else
            bed_file="/n/data1/hms/dbmi/farhat/Sanjana/H37Rv/RLC_Regions.Plus.LowPmapK50E4.H37Rv.ExtendSingleNucs.bed"
        fi    
    
        # get SNPs and MNPs only, bgzip to use bedtools, remove Max's excluded regions, then write to a new file
        # bcftools view --types snps,mnps /n/data1/hms/dbmi/farhat/rollingDB/genomic_data/$sample_ID/pilon/${sample_ID}_full.vcf.gz | bgzip -c | bedtools intersect -a "-" -b $bed_file -v -header > $out_dir/$sample_ID.vcf
        bcftools view --types snps,mnps /n/data1/hms/dbmi/farhat/rollingDB/genomic_data/$sample_ID/pilon/${sample_ID}_full.vcf.gz | bgzip -c | bedtools subtract -a "-" -b $bed_file > $out_dir/$sample_ID.vcf

    fi    
    
done < "$input_file"