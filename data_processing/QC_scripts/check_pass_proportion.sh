#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-02:00
#SBATCH -p short 
#SBATCH --mem=1G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

source activate bioinformatics

# Set the input file name
input_file=$1
output_file=$2
vcf_dir=$3
START=$4
END=$5

thresh=0.75
# vcf_dir="/n/scratch3/users/s/sak0914/annotated_VCF"
# vcf_dir="/n/data1/hms/dbmi/farhat/Sanjana/MIC_data/MIC_ML/VCF"

if ! [ $# -eq 5 ]; then
    echo "Please pass in 5 command line arguments: input file, output file, directory where VCF files are stored, and the start and end coordinates (inclusive)"
    exit
fi


# if the output file exists, delete it so that the new data is not appended to it
if [ -f $output_file ]; then
    rm $output_file
    echo "deleted existing output file"
fi


# Loop through each line of the file
while IFS=",", read -r line; do
    
    # Extract the first column
    sample_ID=$(echo "$line" | cut -d ',' -f 1)
    
    # ignore header, which is ROLLINGDB_ID
    if [ "$sample_ID" != "ROLLINGDB_ID" ]; then
        
        # if echo "$vcf_dir" | grep -q "$search_string"; then
            # num_pass=$(bcftools filter -i "POS >= $START & POS <= $END & (FILTER='PASS' || FILTER='Amb')" $vcf_dir/$sample_ID/pilon/$sample_ID.vcf | awk '$1 !~ /^#/' | wc -l)        
            # num_var=$(bcftools filter -i "POS >= $START & POS <= $END" $vcf_dir/$sample_ID/pilon/$sample_ID.vcf | awk '$1 !~ /^#/' | wc -l)
        # else
        
        num_pass=$(bcftools filter -i "POS >= $START & POS <= $END & (FILTER='PASS' || FILTER='Amb')" $vcf_dir/$sample_ID/pilon/$sample_ID.vcf | awk '$1 !~ /^#/' | wc -l)        
        num_var=$(bcftools filter -i "POS >= $START & POS <= $END" $vcf_dir/$sample_ID/pilon/$sample_ID.vcf | awk '$1 !~ /^#/' | wc -l)
        # fi

        prop_pass=$(echo "scale=4; $num_pass / $num_var" | bc)
        
        # write the proportion that are PASS or Amb ONLY (don't include things like LowCov,Amb in the count)
        # if there are no variants in the region of interest, then an empty string is written
        echo "$sample_ID $prop_pass" >> $output_file
    fi

done < "$input_file"