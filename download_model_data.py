import os
import urllib


def download_model_specific_data(data_dest):
    github_dir = 'https://raw.githubusercontent.com/jianglikun/DeepTTC/main/ESPF'
    names = ['drug_codes_chembl_freq_1500.txt', 'file_list.txt', 'subword_units_map_chembl_freq_1500.csv']

    data_dest = os.path.join(data_dest, 'ESPF')
    if not os.path.exists(data_dest):
        os.mkdir(data_dest)

    for name in names:
        src = os.path.join(github_dir, name)
        dest = os.path.join(data_dest, name)
        urllib.request.urlretrieve(src, dest)