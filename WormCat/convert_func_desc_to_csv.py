import pandas as pd

def convert_functional_description_to_csv(from_file_nm, to_file_nm):
    # Read the input file
    with open(from_file_nm, 'r') as file:
        lines = file.readlines()

    # Initialize empty lists to store data
    data = {'Wormbase_Id': [],
            'gene_nm': [],
            'sequence_id': [],
            'concise_description': [],
            'automated_description': [],
            'gene_class_description': []}

    # Process lines
    index = 0
    index += 1 #Skip line
    index += 1 #Skip line
    index += 1 #Skip line
    while index < len(lines):    
        #print(f"Start {index}:: {lines[index]}")
        wormbase_id, gene_nm, sequence_id = lines[index].strip().split('\t')
        index += 1

        concise_description = ""
        lines[index] =lines[index].replace("Concise description:", "").strip()
        while not lines[index].startswith("Automated description:"):
            #print(f"Concise {index}:: {lines[index]}")
            concise_description += lines[index].strip() + " "
            index += 1

        automated_description = ""
        lines[index] =lines[index].replace("Automated description:", "").strip()
        while not lines[index].startswith("Gene class description:"):
            #print(f"Automated {index}:: {lines[index]}")
            automated_description += lines[index].strip() + " "
            index += 1

        gene_class_description = ""
        lines[index] =lines[index].replace("Gene class description:", "").strip()
        while not lines[index].startswith("="):
            #print(f"Gene class {index}:: {lines[index]}")
            gene_class_description += lines[index].strip() + " "
            index += 1

        index += 1 # Skip =

        data['Wormbase_Id'].append(wormbase_id)
        data['gene_nm'].append(gene_nm)
        data['sequence_id'].append(sequence_id)
        data['concise_description'].append(concise_description.strip())
        data['automated_description'].append(automated_description.strip())
        data['gene_class_description'].append(gene_class_description.strip())

    # Create the DataFrame
    functional_descriptions_df = pd.DataFrame(data)

    functional_descriptions_df.to_csv(to_file_nm, index=False)