#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-02:00
#SBATCH -p short
#SBATCH --mem=5G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics

input_file="/home/sak0914/all_samples_MIC_models.csv"
out_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/resR_variants"

# Check if the output directory exists. If not, raise an error
if [ ! -d "$out_dir" ]; then
    echo "Output directory $out_dir doesn't exist!"
    exit 1
fi

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$',', read -r sample_ID
do
# for vcf_fName in /n/data1/hms/dbmi/farhat/rollingDB/genomic_data/*/pilon/*_full_vcf.gz; do

    if [ ! -f "$out_dir/$sample_ID.vcf.gz" ]; then
    
        og_fName="/n/data1/hms/dbmi/farhat/rollingDB/genomic_data/$sample_ID/WHO_resistance/${sample_ID}_variants_combinedCodons.eff.vcf"

        if [ -f $og_fName ]; then
        
            bcftools filter -i "POS >= 2074751 & POS <= 2075518" $og_fName > "$out_dir/$sample_ID.vcf"
            echo "Finished writing variants file for $sample_ID"
        
        else
    
            TRUST_og_fName="/n/data1/hms/dbmi/farhat/rollingDB/TRUST/Illumina_culture_WGS_processed/$sample_ID/WHO_resistance/${sample_ID}_variants_combinedCodons.eff.vcf"
            
            bcftools filter -i "POS >= 2074751 & POS <= 2075518" $TRUST_og_fName > "$out_dir/$sample_ID.vcf"
            echo "Finished writing variants file for $sample_ID"
        fi

        if [ -f "$out_dir/$sample_ID.vcf" ]; then
            # bgzip and indexing the VCF file are required before running bcftools merge
            bgzip "$sample_ID.vcf"
            bcftools index "$sample_ID.vcf.gz"
        fi

    fi
    
done < "$input_file"