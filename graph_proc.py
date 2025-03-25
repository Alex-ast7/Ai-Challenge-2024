import torch
import numpy as np
import gc
import networkx as nx

idx2relation = [
 'antonym',
 'at location',
 'capable of',
 'causes',
 'created by',
 'is a',
 'desires',
 'has subevent',
 'part of',
 'has context',
 'has property',
 'made of',
 'not capable of',
 'not desires',
 'receives action',
 'related to',
 'used for'
]

class GraphProc():
    def __init__(self, maper, graph, idx2relation=idx2relation, max_edges=64):
        self.maper = maper
        self.G = graph
        self.idx2relation = idx2relation
        self.max_edges = max_edges
        
    def get_rel_concepts(self, cp1, concepts1):
        rels = []
        no_rels = []
        for cp2 in concepts1:
            if cp2 != cp1:
                edge_data = self.G.get_edge_data(cp1,cp2)
                if edge_data:
                    for i in range(len(self.idx2relation)):
                        if edge_data['relation'][i] == 1:
                            rels.append(f"{self.maper[cp1].replace('_',' ')} {self.idx2relation[i].replace('_',' ')} {self.maper[cp2]}")
                else:
                    no_rels.append(self.maper[cp2].replace('_',' '))
        no_rels = f"{self.maper[cp1].replace('_',' ')}  has no relation with {'; '.join(no_rels)}"
        return [no_rels] + rels
    
    def get_concepts_rel_matrix(self, concepts):
        rels = []
        for cp1 in concepts:
            rels.extend(self.get_rel_concepts(cp1,concepts))
        return rels

class GraphProcessorV2():
    def __init__(self, G, embeds, n_hop=1,pad_to_nodes=64):
        self.G = G
        self.embeds = embeds
        self.pad_to_nodes = pad_to_nodes
        self.n_hop = n_hop
        
    def get_k_hop_neighbours(self,node):
        if self.n_hop == 1:
            return list(self.G.neighbors(node))
        nbrs = set([node])
        for l in range(self.n_hop):
            nbrs |= set((nbr for n in nbrs for nbr in self.G.neighbors(n)))
        return list(nbrs)
    
    def process(self,nodes):
        if len(nodes) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            for n in nodes:
                nodes += self.get_k_hop_neighbours(n)
                if len(nodes) >= self.pad_to_nodes:
                    break
        if len(nodes) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            nodes += [0] * (self.pad_to_nodes - len(nodes))
            
        inv_nodes = {n:i for i,n in enumerate(nodes)}    
        sub_graph = self.G.subgraph(nodes)
        edge_attrs,edge_index = [], []
        for node1,node2,edge_feat in sub_graph.edges(data='relation'):
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
        nodes = [n if n < 799272 else 160892 for n in nodes]
        return {
            'edge_index': torch.tensor(edge_index).T,
            'edge_attr': torch.tensor(edge_attrs),
            'x': torch.from_numpy(np.stack(self.embeds[nodes])).to(torch.float)
        }

class GraphProcessorV3():
    def __init__(self, G, embeds, n_hop=1,pad_to_nodes=64):
        self.G = G
        self.embeds = embeds
        self.pad_to_nodes = pad_to_nodes
        self.n_hop = n_hop
        
    def get_k_hop_neighbours(self,node):
        if self.n_hop == 1:
            return list(self.G.neighbors(node))
        nbrs = set([node])
        for l in range(self.n_hop):
            nbrs |= set((nbr for n in nbrs for nbr in self.G.neighbors(n)))
        return list(nbrs)
    
    def chek_unique(self,nodes):
        """Unique without sort"""
        indexes = np.unique(nodes, return_index=True)[1]
        return [nodes[i] for i in sorted(indexes)]
    
    def process_image(self, nodes, image_triplets, image_nodes):
        if len(np.unique(nodes+image_nodes)) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            for n in nodes:
                nodes += self.get_k_hop_neighbours(n)
                nodes = self.chek_unique(nodes)
                if len(np.unique(nodes+image_nodes)) >= self.pad_to_nodes:
                    break
        print(len(np.unique(nodes)), len(nodes), len(np.unique(nodes+image_nodes)))
        if len(nodes) >= self.pad_to_nodes:
            pad_length = self.pad_to_nodes - len(np.unique(nodes+image_nodes)) + len(np.unique(nodes))
            nodes = nodes[:pad_length]
        else:
            pad_length = self.pad_to_nodes - len(np.unique(nodes+image_nodes)) + len(np.unique(nodes))
            nodes += [160892] * (pad_length - len(nodes))
        print(len(nodes))
        sub_graph = self.G.subgraph(nodes).copy()
        sub_graph.add_nodes_from(image_nodes)
        inv_nodes = {n:i for i,n in enumerate(sub_graph.nodes)}    
        edge_attrs,edge_index = [], []
        for node1,node2,edge_feat in sub_graph.edges(data='relation'):
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
            
        for node1, edge_feat, node2 in image_triplets:
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
        
        nodes = list(sub_graph.nodes)
        nodes = [n if n < 799272 else 160892 for n in nodes]
        return {
            'edge_index': torch.tensor(edge_index).T,
            'edge_attr': torch.tensor(edge_attrs),
            'x': torch.from_numpy(np.stack(self.embeds[nodes])).to(torch.float)
        }

    
    def process(self, nodes):
        if len(nodes) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            for n in nodes:
                nodes += self.get_k_hop_neighbours(n)
                if len(nodes) >= self.pad_to_nodes:
                    break
        if len(nodes) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            nodes += [0] * (self.pad_to_nodes - len(nodes))
            
        inv_nodes = {n:i for i,n in enumerate(nodes)}    
        sub_graph = self.G.subgraph(nodes).copy()
        edge_attrs,edge_index = [], []
        for node1,node2,edge_feat in sub_graph.edges(data='relation'):
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
        nodes = [n if n < 799272 else 160892 for n in nodes]
        return {
            'edge_index': torch.tensor(edge_index).T,
            'edge_attr': torch.tensor(edge_attrs),
            'x': torch.from_numpy(np.stack(self.embeds[nodes])).to(torch.float)
        }

class GraphProcessorV4():
    def __init__(self, G, embeds, n_hop=1,pad_to_nodes=64):
        self.G = G
        self.embeds = embeds
        self.pad_to_nodes = pad_to_nodes
        self.n_hop = n_hop
        
    def get_k_hop_neighbours(self,node):
        if self.n_hop == 1:
            return list(self.G.neighbors(node))
        nbrs = set([node])
        for l in range(self.n_hop):
            nbrs |= set((nbr for n in nbrs for nbr in self.G.neighbors(n)))
        return list(nbrs)
    
    def chek_unique(self,nodes):
        """Unique without sort"""
        indexes = np.unique(nodes, return_index=True)[1]
        return [nodes[i] for i in sorted(indexes)]
    
    def process_image(self, nodes, image_triplets, image_nodes):
        if len(np.unique(nodes+image_nodes)) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            for n in nodes:
                nodes += self.get_k_hop_neighbours(n)
                nodes = self.chek_unique(nodes)
                if len(np.unique(nodes+image_nodes)) >= self.pad_to_nodes:
                    break
        nodes = self.chek_unique(nodes)
        image_nodes = self.chek_unique(image_nodes)
        nodes = [n for n in nodes if n not in image_nodes]
        image_nodes_uniq_len = len([n for n in image_nodes if n not in nodes])
        
        if len(nodes) >= self.pad_to_nodes - image_nodes_uniq_len:
            nodes = nodes[:self.pad_to_nodes - image_nodes_uniq_len]
            nodes += [n for n in image_nodes if n not in nodes]
        else:
            nodes += [n for n in image_nodes if n not in nodes]
            nodes += [160892] * (self.pad_to_nodes - len(nodes))
        sub_graph = self.G.subgraph(nodes).copy()
        inv_nodes = {n:i for i,n in enumerate(sub_graph.nodes)}    
        edge_attrs,edge_index = [], []
        for node1,node2,edge_feat in sub_graph.edges(data='relation'):
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
            
        for node1, edge_feat, node2 in image_triplets:
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
        if not edge_attrs:
            edge_index.append([0, 1])
            edge_attrs.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        nodes = [n if n < 799272 else 160892 for n in nodes]
        return {
            'edge_index': torch.tensor(edge_index).T,
            'edge_attr': torch.tensor(edge_attrs),
            'x': torch.from_numpy(np.stack(self.embeds[nodes])).to(torch.float)
        }

    
    def process(self, nodes):
        nodes = self.chek_unique(nodes)
        if len(nodes) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            for n in nodes:
                nodes += self.get_k_hop_neighbours(n)
                nodes = self.chek_unique(nodes)
                if len(nodes) >= self.pad_to_nodes:
                    break
        
        if len(nodes) >= self.pad_to_nodes:
            nodes = nodes[:self.pad_to_nodes]
        else:
            nodes += [160892] * (self.pad_to_nodes - len(nodes))
            
        sub_graph = self.G.subgraph(nodes).copy()
        inv_nodes = {n:i for i,n in enumerate(nodes)}   
        edge_attrs,edge_index = [], []
        for node1,node2,edge_feat in sub_graph.edges(data='relation'):
            edge_index.append([inv_nodes[node1], inv_nodes[node2]])
            edge_attrs.append(edge_feat)
        if not edge_attrs:
            edge_index.append([0, 1])
            edge_attrs.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        nodes = [n if n < 799272 else 160892 for n in nodes]
        return {
            'edge_index': torch.tensor(edge_index).T,
            'edge_attr': torch.tensor(edge_attrs),
            'x': torch.from_numpy(np.stack(self.embeds[nodes])).to(torch.float)
        }

