import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdchem import BondType
from rdkit import RDLogger
import pickle as pkl
RDLogger.DisableLog('rdApp.*')



class MyData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, fingerPrint=None, bertEmbedding = None,
                 gexpr=None, mutation=None, methylation=None, drugId = None, cellId = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, **kwargs)

        # 添加额外属性
        self.gexpr = gexpr
        #self.mutation = mutation
        self.methylation = methylation
        self.fingerPrint = fingerPrint
        self.bertEmbedding = bertEmbedding
        self.drugId = drugId
        self.cellId = cellId


class MyDataset(Dataset):
    def __init__(self,gexpr_feature,mutation_feature,methylation_feature,all_respond,drugid2smiles):
        super().__init__()
        self.gexpr_feature = gexpr_feature
        self.mutation_feature = mutation_feature
        self.methylation_feature = methylation_feature
        self.all_respond = all_respond
        self.smiles_Graph_Dict = {value:key for key,value in drugid2smiles.items()}
        self.smiles_FingerPrint_Dict = {value: key for key, value in drugid2smiles.items()}
        self.smiles_drugid_Dict = {value: key for key, value in drugid2smiles.items()}
        self.smiles_BertEmbedding_Dict = {}
        self.atom_embedding_dict = {}
        self.smiles_embedding_dict = {}
        self.atom_types = [35, 5, 6, 7, 8, 9, 78, 15, 16, 17, 53]
        self.h_count_stat = [0] * 5  # 初始化计数器，记录 one-hot 中每个位置为1的次数
        with open("./chemberta_embeddings.pkl", 'rb') as f:
            self.smiles_BertEmbedding_Dict = pkl.load(f)
        with open("./data_process/atom_embedding_dict.pkl", 'rb') as f:
            self.atom_embedding_dict = pkl.load(f)
        with open('./data_process/smiles_embedding_dict.pkl', 'rb') as f:
            self.smiles_embedding_dict = pkl.load(f)
        for s in self.smiles_Graph_Dict:
            self.smiles_Graph_Dict[s] = self.smiles_to_graph(s)
        for s in self.smiles_FingerPrint_Dict:
            self.smiles_FingerPrint_Dict[s] = self.smiles_to_fingerPrint(s)


    def smiles_to_fingerPrint(self,smiles):
        USE_MOIR_EMBEDDING = False
        USE_MACCSKEY = False
        USE_RDKitKEY = True
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string: "+ smiles)

        if USE_MOIR_EMBEDDING:
            #print("use MOIR embedding")
            fingerprint = self.smiles_embedding_dict[smiles].tolist()
            return fingerprint
        elif USE_MACCSKEY:
            #print("use MACCSKEY")
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            fingerprint_array = fingerprint.ToBitString()
            fingerprint_list = [int(bit) for bit in fingerprint_array]
            return fingerprint_list
        elif USE_RDKitKEY:
            #print("use MorganFingerprint")
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol ,3 ,nBits=512)
            fingerprint_array = fingerprint.ToBitString()
            fingerprint_list = [int(bit) for bit in fingerprint_array]
            return fingerprint_list

    def _get_hydrogen_one_hot(self, atom):
        num_h = min(atom.GetTotalNumHs(), 3)  # 超过4个H的截断为4
        return [int(num_h == i) for i in range(4)]  # 5维one-hot

    #abondan?, sclace seems better choice by gpt
    def _get_hydrogen_scaled(self, atom):
        num_h = min(atom.GetTotalNumHs(), 3)  # 截断为最多4个氢
        return [num_h / 3.0]

    def _get_ring_size_features(self, mol, atom_idx):
        """计算原子参与的 5,6,7 元环数量（3维向量）"""
        r = range(5, 8)
        ring_info = mol.GetRingInfo()
        ring_counts = [0] * len(r)

        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            if ring_size in r:
                if atom_idx in ring:
                    ring_counts[ring_size - min(r)] += 1
        return ring_counts

    def smiles_to_graph(self,smiles):

        """
        generate "graph" based on smiles with the help of rdkit.Chem
        :return: x,edge_index,edge_attr (tensor)
        """
        USE_MOIR_EMBEDDING = True
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 节点特征
        x = []
        for atom in mol.GetAtoms():
            #if USE_MOIR_EMBEDDING:
                #atom_feature = self.atom_embedding_dict[atom.GetAtomicNum()].tolist()
            #else:
            atom_feature = [0] * len(self.atom_types)
            atom_feature[self.atom_types.index(atom.GetAtomicNum())] = 1  # 设置相应位置为 1

            atom_feature.append(atom.GetTotalNumHs())  # 连接的氢原子数
            atom_feature.append(atom.GetFormalCharge())  # 形式电荷
            atom_feature.append(atom.GetTotalValence())  # 价态
            atom_feature.append(int(atom.GetIsAromatic()))  # 是否为芳香性
            # more features
            atom_feature.append(int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP))  # sp混成
            atom_feature.append(int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2))  # sp2混成
            atom_feature.append(int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3))  # sp3混成
            atom_feature.append(int(atom.IsInRing()))  # 是否在环中
            atom_feature.append(atom.GetNumExplicitHs())  # 显式氢原子数
            atom_feature.append(len([nbr for nbr in atom.GetNeighbors()]))  # 邻居数

            # 新特征
            #atom_feature.extend(self._get_hydrogen_one_hot(atom))
            #atom_feature.extend(self._get_hydrogen_scaled(atom))
            #atom_feature.extend(self._get_ring_size_features(mol ,atom.GetIdx()))

            x.append(atom_feature)

        x = torch.tensor(x, dtype=torch.float)

        # 边索引和边特征
        edge_index = []
        edge_attr = []

        # 获取分子的3D坐标（如果有的话）
        unsuccess_generate3d = AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        if (not unsuccess_generate3d):
            conformer = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        else:
            smi = Chem.MolToSmiles(mol)
            print(smi, "unsucess generate 3d structure")
            conformer = None

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()  # 起始原子索引
            j = bond.GetEndAtomIdx()  # 结束原子索引

            # 添加边索引
            edge_index.append([i, j])
            edge_index.append([j, i])  # 无向图双向添加

            bond_onehot = None
            # 化学键类型
            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                bond_onehot = [1,0,0,0]
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                bond_onehot = [0,1,0,0]
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                bond_onehot = [0,0,1,0]
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                bond_onehot = [0,0,0,1]
            else:
                bond_onehot = [0,0,0,0]

            # 是否芳香键
            is_aromatic = int(bond.GetIsAromatic())

            # 是否在环内
            is_in_ring = int(bond.IsInRing())

            # 电负性差异
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            electronegativity_diff = abs(atom1.GetAtomicNum() - atom2.GetAtomicNum())

            # 化学键长度
            bond_length = 0.0
            if conformer:  # 如果有3D坐标，则计算键长
                pos1 = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
                pos2 = conformer.GetAtomPosition(bond.GetEndAtomIdx())
                bond_length = pos1.Distance(pos2)

            # 化学键手性
            stereo = int(bond.GetStereo())

            # 键强度：根据键类型简单赋值
            bond_strength = 0
            if bond_type == Chem.rdchem.BondType.SINGLE:
                bond_strength = 1
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                bond_strength = 2
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                bond_strength = 3
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                bond_strength = 1  # 芳香性通常表现为相对较弱的键

            # 是否为双键的共轭
            is_conjugated = int(bond.GetIsConjugated())

            # 合并所有特征

            edge_features = [
                is_aromatic,  # 是否芳香键
                is_in_ring,  # 是否在环内
                electronegativity_diff,  # 电负性差异
                stereo,  # 化学键手性
                bond_strength,  # 键强度
                bond_length,  # 键长度
                is_conjugated  # 是否共轭
            ]
            edge_features.extend(bond_onehot)

            # 无向图双向添加
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        # print("edge_index:",edge_index.shape," edge_attr:",edge_attr.shape)
        #print(smiles, " >> generate graph with x:", x.shape)
        return [x, edge_index, edge_attr]

    def __len__(self):
        return len(self.all_respond)

    def __getitem__(self, idx):
        cell_line = self.all_respond[idx][0]
        smiles = self.all_respond[idx][1]
        data = self.smiles_Graph_Dict[smiles]
        fp = self.smiles_FingerPrint_Dict[smiles]
        bertEmbedding = self.smiles_BertEmbedding_Dict[smiles]
        y = torch.tensor([self.all_respond[idx][2]],dtype=torch.float).unsqueeze(0)
        drugId = self.smiles_drugid_Dict[smiles]
        #x,edge_index,edge_attr = self.smiles_to_graph(smiles)

        # mutation = torch.tensor(self.mutation_feature.loc[cell_line].tolist(), dtype=torch.int),
        return MyData(x=data[0], edge_index=data[1],edge_attr=data[2], y=y,
                      fingerPrint=torch.tensor(fp,dtype=torch.float),
                      bertEmbedding=torch.tensor(bertEmbedding,dtype=torch.float),
                      gexpr=torch.tensor(self.gexpr_feature.loc[cell_line].tolist(),dtype=torch.float),
                      methylation=torch.tensor(self.methylation_feature.loc[cell_line].tolist(),dtype=torch.float),
                      drugId=drugId, cellId=cell_line)




