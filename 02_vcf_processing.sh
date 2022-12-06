# have a separate environment for bcftools
source activate sambcftools

# the 1 is the first additional command line argument
cat "$1" | while read vcf_file; do
    
    # remove the full path
    fName=$(basename $vcf_file)

    # remove the part after the underscore, which leaves only the isolate name remaining
    isolate=${fName%_*}
    
    # check if the filtered VCF file does not exist. If it doesn't, filter and create it
    if [ ! -f "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF/$isolate.vcf" ]; then
    
        # filter the VCF file: keep all variants and all FILTER tags. This step just removes reference calls
        bcftools view --types snps,indels,mnps,other "$vcf_file" > "/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF/$isolate.vcf"
    fi
    
done


# snpEff is in the base environment
source deactivate

# navigate to the directory with all VCF files
cd /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF

# create a text with the file names to use with snpEff
ls -d $PWD/* > /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_list.txt

# annotate all VCF files in the VCF directory (because this step is so fast, it can be done on old files, even though it's redundant)
snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/vcf_files_list.txt