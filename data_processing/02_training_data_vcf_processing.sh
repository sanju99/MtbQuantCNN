#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-02:00
#SBATCH -p short 
#SBATCH --mem=2G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# sbatch data_processing/02_training_data_vcf_processing.sh /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/KAN/paths.txt /n/scratch3/users/s/sak0914/annotated_VCF

source activate bioinformatics

# command line arguments:

# 1. text file with all original VCF files to read from and extract variants from
# 2. directory to store filtered VCF files. DON'T INCLUDE A SLASH AT THE END OF IT --> use a scratch directory because files get copied

if ! [ $# -eq 2 ]; then
    echo "Please pass in 2 command line arguments: a text file with the paths to the full gzipped VCF files and the directory to store VCF files of variants only"
    exit
fi

cat "$1" | while read vcf_file; do
    
    # remove the full path
    fName=$(basename $vcf_file)

    # remove the part after the underscore, which leaves only the isolate name remaining
    isolate=${fName%_*}
    
    # check if the filtered VCF file does not exist. If it doesn't, filter and create it
    if [ ! -f "$2/$isolate.eff.vcf" ]; then
    
        # filter the VCF file: keep all variants and all FILTER tags. This step just removes reference calls
        bcftools view --types snps,indels,mnps,other "$vcf_file" > "$2/$isolate.vcf"
        echo "Created VCF file for $isolate"
    fi
done

# navigate to the directory with all VCF files
cd $2

# exclude files that are already annotated so that SNPEff doesn't get run on them too. The ~+ ensures that we get the full path
# the first part of the line gets all file names that DO NOT have the .eff.vcf extension
find ~+ -type f  ! -name "*.eff.vcf" > /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_to_annot.txt
num_new_files=$(wc -l < /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_to_annot.txt)

# annotate all VCF files in the above text file
# it creates an annotated version of each file at fName.eff.vcf
if (( "$num_new_files" > 0 )); then
    echo "Running snpEff on $num_new_files files"
    snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_to_annot.txt
fi
    
# remove the unannotated files. They take up space and would get re-annotated based on the above logic. This line deletes everything WITHOUT .eff.vcf
find . -type f  ! -name "*.eff.vcf"  -delete

# all remaining steps only need to be run if there are new files
if (( "$num_new_files" > 0 )); then

    for fName in `ls |grep ".vcf"`; do
    
        # get isolate name to check if it's already in the file. The paths in the VCF directory are $vcf_dir/isolate.eff.vcf, so split on the "." character and get the first part
        isolate=$(basename "$fName" | cut -d "." -f 1)
        
        # Use awk to check if the isolate is present in the first column, which is ROLLINGDB_ID
        # include .eff because that is 
        found_isolate=$(awk -F'\t' -v search="$isolate.eff" -v col="1" '$col == search' /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv)
    
        # Check if the output is empty (string not found). If it is empty, then run fast-lineage-caller and add to the existing dataframe
        if [ -z "$found_isolate" ]; then
            # Perform actions when the string is not found
            echo "Getting lineages for $isolate"
            fast-lineage-caller $fName --noheader --pass >> /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/lineages.tsv
        fi
        
    done
    
    # next script requires base environment
    conda deactivate
    
    # make the Coll2014 and Freschi2020 SNP matrices for all isolates
    python3 -u /home/sak0914/MtbQuantCNN/data_processing/03_make_lineage_matrix.py coll $2
    python3 -u /home/sak0914/MtbQuantCNN/data_processing/03_make_lineage_matrix.py freschi $2
fi