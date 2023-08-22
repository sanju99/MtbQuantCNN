import subprocess, sys, os, warnings, glob
import pandas as pd
warnings.filterwarnings("ignore")

_, sample_id_txt_file = sys.argv

sample_ids = pd.read_csv(sample_id_txt_file, sep="\t", header=None)[0].values
print(f"Downloading data for {len(sample_ids)} samples\n")

fastq_dir = "/n/data1/hms/dbmi/farhat/rollingDB/fastq_db"

for sample_id in sample_ids:

    for file in glob.glob(os.path.join(fastq_dir, f"{sample_id}/*.fastq.gz")):
        print(f"Removed existing file {file}")
        os.remove(file)
    
    proc = subprocess.Popen(f"(wget -q -O - ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{sample_id[:6]}/ | grep -o '00[0-9]' | sort -u)", shell=True, encoding='utf8', stdout=subprocess.PIPE)
    
    # remove trailing newline character. newline separates directories
    output = proc.communicate()[0].rstrip("\n")
    dirs_lst = output.split("\n")
    print(dirs_lst)
    
    FQ1 = os.path.join(fastq_dir, sample_id, f"{sample_id}_R1.fastq.gz")
    FQ2 = os.path.join(fastq_dir, sample_id, f"{sample_id}_R2.fastq.gz")
    
    # Iterate through directory names and construct full FTP paths
    for possible_dir in dirs_lst:
        
        possible_R1=f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{sample_id[:6]}/{possible_dir}/{sample_id}/{sample_id}_1.fastq.gz"
        possible_R2=f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{sample_id[:6]}/{possible_dir}/{sample_id}/{sample_id}_2.fastq.gz"
        print(f"Trying {possible_R1}")

        # try downloading data. If it's not successful, remove the file to continue on to the next possible directory
        subprocess.run(f"wget -q --show-progress {possible_R1} -O {FQ1} || rm -f {FQ1}", shell=True)
        subprocess.run(f"wget -q --show-progress {possible_R2} -O {FQ2} || rm -f {FQ2}", shell=True)
    
        if os.path.isfile(FQ1) and os.path.isfile(FQ2):
            print(f"Finished downloading FASTQ files for {sample_id}!")

            proc = subprocess.Popen(f"gunzip -c {FQ1} | wc -l", shell=True, encoding='utf8', stdout=subprocess.PIPE)
            FQ1_line_count = int(proc.communicate()[0].rstrip("\n"))
    
            proc = subprocess.Popen(f"gunzip -c {FQ2} | wc -l", shell=True, encoding='utf8', stdout=subprocess.PIPE)
            FQ2_line_count = int(proc.communicate()[0].rstrip("\n"))

            assert FQ1_line_count == FQ2_line_count
            assert FQ1_line_count > 0
            break