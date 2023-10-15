#!/bin/bash 
#SBATCH -c 1
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=1G
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

if ! [ $# -eq 2 ]; then
    echo "Please pass in 2 command line arguments: input file, output_file"
    exit
fi

# command line arguments
input_file=$1
output_file=$2

# Read the TSV file line by line, skiping the header. IFS sets the field separator. Here, it is tab
while IFS=$'\t', read -r sample_ID
do    

    fastq1_file="/n/data1/hms/dbmi/farhat/rollingDB/fastq_db/$sample_ID/${sample_ID}_R1.fastq.gz"
    fastq2_file="/n/data1/hms/dbmi/farhat/rollingDB/fastq_db/$sample_ID/${sample_ID}_R2.fastq.gz"

    # check that the original FASTQ files have the same numbers of lines
    FQ1_line_count=$(gunzip -c $fastq1_file | wc -l)
    FQ2_line_count=$(gunzip -c $fastq2_file | wc -l)
    
    # check that neither FASTQ file has no reads
    if [ $FQ1_line_count -eq 0 ] || [ $FQ2_line_count -eq 0 ]; then
        echo "Error: $fastq1_file or $fastq2_file has no reads"
        echo $sample_ID >> $output_file
    # Compare the counts and raise an error if they are not equal 
    elif [ "$FQ1_line_count" -ne "$FQ2_line_count" ]; then
        echo "Error: $fastq1_file and $fastq2_file have different line counts"
        echo $sample_ID >> $output_file
    else
        echo "Line counts in paired-end FASTQ files for $sample_ID match"
    fi

done < "$input_file"