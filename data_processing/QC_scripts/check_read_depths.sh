#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-04:00
#SBATCH -p short 
#SBATCH --mem=25G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

# "samples.csv"
input_file=$1
output_file=$2

thresh=10

if ! [ $# -eq 2 ]; then
    echo "Please pass in 2 command line arguments: input file and output file"
    exit
fi

# Loop through each line of the file
while IFS=",", read -r line; do
    
    # Extract the first column
    sample_ID=$(echo "$line" | cut -d ',' -f 1)
    
    # ignore header, which is ROLLINGDB_ID
    if [ "$sample_ID" != "ROLLINGDB_ID" ]; then
    
        if [ -f "/n/scratch3/users/s/sak0914/MIC_ML/$sample_ID/pilon/${sample_ID}_full.vcf.gz" ]; then

            min_depth=$(gunzip -c "/n/scratch3/users/s/sak0914/MIC_ML/$sample_ID/pilon/${sample_ID}_full.vcf.gz" | bcftools query -f '%DP\n' | sort -n | uniq | tail -n 1)
            #min_depth=$(bcftools query -f '%DP\n' "/n/scratch3/users/s/sak0914/MIC_ML/$sample_ID/pilon/${sample_ID}.vcf" | sort -n | uniq | tail -n 1)

            # check if value1 is less than value2
            if [ "$min_depth" -lt "$thresh" ]; then
                echo "$sample_ID" >> "$output_file"
            else
                echo "$sample_ID"
            fi        
        fi
   fi
    
done < "$input_file"