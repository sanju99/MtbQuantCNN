#!/bin/bash 
#SBATCH -c 10
#SBATCH -t 0-11:59 
#SBATCH -p short
#SBATCH --mem=30G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# # navigate to the directory with all VCF files
# # cd /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF
# cd /n/scratch3/users/s/sak0914/vcf_for_annot

# # create a text with the file names to use with snpEff
# # ls -d $PWD/*.vcf > /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_list.txt
# ls -d $PWD/*.vcf > /n/scratch3/users/s/sak0914/vcf_files_list.txt

# annotate all VCF files in the VCF directory (because this step is so fast, it can be done on old files, even though it's redundant)
# snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_list.txt
# snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList /n/scratch3/users/s/sak0914/vcf_files_list.txt

# if the lineage file exists, delete it so that the new lineages are not appended to the same file
if [ -f /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv ]; then
    rm /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv
    echo "deleted existing lineage file"
fi

# run fast-lineage-caller to update the lineages file. Use only variants with the PASS flag
for file in /n/scratch3/users/s/sak0914/vcf_for_lineage/*.vcf; do
    fast-lineage-caller "$file" --noheader --pass >> /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv
done