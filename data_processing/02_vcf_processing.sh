#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=5G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# submit with sbatch data_processing/02_vcf_processing.sh /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/RIF/paths.txt /n/scratch3/users/s/sak0914/annotated_VCF

# have a separate environment for bcftools
source activate bioinformatics

# command line arguments:

# 1. text file with all original VCF files to read from and extract variants from
# 2. directory to store filtered VCF files. DON'T INCLUDE A SLASH AT THE END OF IT --> use a scratch directory because files get copied

cat "$1" | while read vcf_file; do
    
    # remove the full path
    fName=$(basename $vcf_file)

    # remove the part after the underscore, which leaves only the isolate name remaining
    isolate=${fName%_*}
    
    # check if the filtered VCF file does not exist. If it doesn't, filter and create it
    if [ ! -f "$2/$isolate.vcf" ]; then
    
        # filter the VCF file: keep all variants and all FILTER tags. This step just removes reference calls
        bcftools view --types snps,indels,mnps,other "$vcf_file" > "$2/$isolate.vcf"
    else
        echo "$2/$isolate.vcf exists"
    fi
done

# conda deactivate --> no longer needed because everything is in the same environment

# navigate to the directory with all VCF files
cd $2

# create a text file with the names of all files we want to annotate
ls -d $PWD/* > /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_list.txt

# annotate all VCF files in the VCF directory (because this step is so fast, it can be done on old files, even though it's redundant)
# it creates an annotated version of each file at fName.eff.vcf
snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_list.txt

# if the lineage file exists, delete it so that the new lineages are not appended to the same file
if [ -f /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv ]; then
    rm /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv
    echo "deleted existing lineage file"
fi

# # run fast-lineage-caller to update the lineages file. Use only variants with the PASS flag. The filtered VCFs will have all variants, including low-quality ones
# for file in "$2/*.vcf"; do
#     fast-lineage-caller "$file" --noheader --pass >> /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv
# done

# for i in `ls |grep ".vcf"`; do
#     fast-lineage-caller genomic_data/SAMEA968141.vcf --noheader
# done >> results.tsv