import numpy as np


def get_all_proteins(dataset_name):
    proteins = []
    types = {"train", "val", "test"}
    for type in types:
        data_file = "data/" + dataset_name + "/" + dataset_name + "_" + type + ".txt"
        with open(data_file, "r") as f:
            lines = f.readlines()
            for i in lines:
                proteins.append(i.split(" ")[1])
    return proteins


def seq_to_kmers(seq, k):
    N = len(seq)
    return [seq[i:i + k] for i in range(N - k + 1)]


def get_k_gram_seqs(proteins, k):
    pros = []
    for protein in proteins:
        pro2 = ""
        for i in protein:
            pro2 += i

        pros.append(seq_to_kmers(pro2, k=k))
    return pros


def get_k_gram_seq(protein, k):
    pro = ""
    for i in protein:
        pro += i
    pro = seq_to_kmers(pro, k=k)
    return pro


def get_dict(proteins):
    num = 1
    dict = {}
    for seq in proteins:
        line = []
        for i in seq:
            if i not in dict:
                dict[i] = num
                num += 1
            line.append(dict[i])
    return dict


def k_gram_save(data_file, k, save_name):
    with open(data_file, 'r') as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""  # The '.' represents multiple chemical molecules
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    print("Protein----dataset_size: ", N)
    proteins = []
    num_max = 0

    len_max = 0
    len_sum = 0
    len_num = 0
    for i, data in enumerate(data_list):
        lis = []
        protein = data.strip().split(" ")[1]
        pro = get_k_gram_seq(protein, k)
        for seq in pro:
            lis.append(dict[seq])
        len_max = max(len(lis), len_max)
        len_num += 1
        len_sum += len(lis)
        if len(lis) > dim:
            lis = lis[0:dim]
        while len(lis) < dim:
            lis.append(0)

        print(len(lis))
        num_max = max(max(lis), num_max)
        proteins.append(lis)

    print("num_max: ", num_max)
    print("len_max: ", len_max)
    print("len_avg: ", len_sum / len_num)

    np.save(save_name, proteins)


if __name__ == '__main__':
    dataset_name = "Davis"
    k = 1
    dim = 256
    proteins = get_k_gram_seqs(get_all_proteins(dataset_name), k)
    dict = get_dict(proteins)
    print(dict)

    types = {"train", "val", "test"}
    for type in types:
        data_file = "data/" + dataset_name + "/" + dataset_name + "_" + type + ".txt"
        save_name = "data/" + dataset_name + "/input/" + dataset_name + "_" + type + "_proteins_" + str(k) + "_gram"
        k_gram_save(data_file, k=k, save_name=save_name)
