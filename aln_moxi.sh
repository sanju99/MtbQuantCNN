#!/bin/bash 
#SBATCH -c 2
#SBATCH -t 0-11:59
#SBATCH -p short 
#SBATCH --mem=30G 
#SBATCH -o /home/sak0914/Errors/zerrors_%j.out 
#SBATCH -e /home/sak0914/Errors/zerrors_%j.err 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skulkarni@g.harvard.edu

perl data_processing/snpConcatenater_w_exclusion_frompilonvcf_2.9_edit_2022.pl /n/data1/hms/dbmi/farhat/Sanjana/MIC_data/single_drugs/MOXI/filt_paths.txt /n/data1/hms/dbmi/farhat/mchen/exclude.BED /n/data1/hms/dbmi/farhat/tb_cnn/IDfail.tab INDEL REGION 4997-9818 pos > gyrBA.fasta