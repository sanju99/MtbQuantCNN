set -o errexit # any error will cause the shell script to exit immediately. This is not native bash behavior
source activate bioinformatics

# command line arguments
# input_file="./VCFs_remove_duplicate_indels.csv"
# out_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/VCF_clean"

input_file="/home/sak0914/transmission_lineages/VCFs_remove_duplicate_indels_more.csv"
out_dir="/n/data1/hms/dbmi/farhat/Sanjana/transmission_analysis/VCF"

# Check if the output directory exists. If not, raise an error
if [ ! -d "$out_dir" ]; then
    echo "Output directory $out_dir doesn't exist!"
    exit 1
fi

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$',', read -r sample_ID fName exclude_pos
do
    # remove the second variant (the structural variant)
    bcftools filter -i "$exclude_pos" $fName > "$out_dir/$sample_ID.vcf"
    
done < "$input_file"