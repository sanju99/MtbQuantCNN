#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=25G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate bioinformatics

input_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/MFS_IDs_bam_files.txt"
vcf_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/VCF"
ref_genome="/n/data1/hms/dbmi/farhat/mtb_data/h37rv/h37rv.fna"

fNames=($(cat "${input_file}"))
echo "Making VCF files for ${#fNames[@]} isolates"

while IFS=$'\t', read -r bam_file
do
    sample_ID=$(basename "$bam_file" | cut -d "." -f 1)

    if [ ! -f "$vcf_dir/$sample_ID.vcf" ]; then

        pilon -Xmx18g --genome "$ref_genome" --bam $bam_file --output "$sample_ID" --outdir "$vcf_dir" --variant
        
        bcftools view --types snps,indels,mnps,other "$vcf_dir/$sample_ID.vcf" > "$vcf_dir/${sample_ID}_small.vcf"
    
        rm "$vcf_dir/$sample_ID.vcf"
        mv "$vcf_dir/${sample_ID}_small.vcf" "$vcf_dir/$sample_ID.vcf"
        rm "$vcf_dir/$sample_ID.fasta"

    else
        echo "$vcf_dir/$sample_ID.vcf" already exists
    fi

done < $input_file

isolates_file="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/samples_full_paths.txt"
find $vcf_dir -type f > $isolates_file
fNames_annot=($(cat "${isolates_file}"))
echo "Annotating variants with snpEff for ${#fNames_annot[@]} isolates"

snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList -no-downstream -no-upstream $isolates_file