vcf_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/TRUST/VCF"
isolates_file='/home/sak0914/MtbQuantCNN/analysis/TRUST/samples_to_annot.txt'

while IFS=$'\t', read -r MFS_ID
do
    sample_id=$(basename "$MFS_ID" | cut -d "." -f 1)
    new_fName=$vcf_dir/$sample_id.vcf
    bcftools view --types snps,indels,mnps,other $full_VCF_fName > $new_fName

done < $input_file

find $vcf_dir -maxdepth 1 -type f > $isolates_file
fNames=($(cat "${isolates_file}"))
echo "Annotating ${#fNames[@]} isolates"

# snpEff eff Mycobacterium_tuberculosis_gca_000195955 -noStats -fileList -no-downstream -no-upstream $isolates_file