#This script is the implementation of my custom tree to calculate the wordnet similarity, based on https://github.com/MadryLab/robustness/blob/master/robustness/tools/imagenet_helpers.py

import os
import numpy as np
import json
import math

class Node():
    '''
    Class for representing a node in the ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, wnid, name=""):
        """
        Args:
            wnid (str) : WordNet ID for synset represented by node
            parent_wnid (str) : WordNet ID for synset of node's parent
            name (str) : word/human-interpretable description of synset 
        """

        self.wnid = wnid
        self.name = name
        self.parents_wnid = []
        self.children_wnid_directly = []
        self.children_wnid_all = []
        self.contained_labels = set()
        self.class_num = -1

    def change_class_num(self, class_num):
        self.class_num = class_num
    
    def add_child_directly(self, child):
        """
        Add child to given node.

        Args:
            child (Node) : Node object for child
        """
        self.children_wnid_directly.append(child)

    def add_children_all(self, child):
        
        self.children_wnid_all.append(child)
    
    def add_parent(self, parent):
        """
        Add child to given node.

        Args:
            child (Node) : Node object for child
        """
        self.parents_wnid.append(parent)
    def add_contained_labels(self, wnid, number):
        self.contained_labels.add((wnid,number))
    
    def get_parents(self):
        return self.parents_wnid
    
    def get_children_direct(self):
        return self.children_wnid_directly
    
    def get_children_all(self):
        return self.children_wnid_all
    
    def get_wnid(self):
        return self.wnid
    
    def get_name(self):
        return self.name
    
    def get_class_num(self):
        return self.class_num
    
    def get_contained_labels(self):
        return self.contained_labels
    
    def __str__(self):
        return f'Name: ({self.name}), Wnid: ({self.wnid}), Num of Parents: ({len(self.parents_wnid)}),  Num of Children: ({len(self.children_wnid_directly)})'
    
    def __repr__(self):
        return f'Name: ({self.name}), Wnid: ({self.wnid}), Num of Parents: ({len(self.parents_wnid)}),  Num of Children: ({len(self.children_wnid_directly)})'

class ImagenetTree():
    
    def __init__(self, ds_path, ds_info_path):
        self.tree = {}
        
        ret = self.load_imagenet_info(ds_path, ds_info_path)
        self.in_wnids, self.wnid_to_name, self.wnid_to_num, self.num_to_name = ret
            
        with open(os.path.join(ds_info_path, 'wordnet.is_a.txt'), 'r') as f:
            for line in f.readlines():
                parent_wnid, child_wnid = line.strip('\n').split(' ')
                parentNode = self.get_node(parent_wnid)
                childNode = self.get_node(child_wnid)
                parentNode.add_child_directly(childNode)
                childNode.add_parent(parentNode)

        for wnid in self.in_wnids:
            self.tree[wnid].change_class_num(self.wnid_to_num[wnid])

        self.add_all_labels()

        del_nodes = [wnid for wnid in self.tree \
                     if (len(self.tree[wnid].contained_labels) == 0 and self.tree[wnid].class_num == -1)]
        for d in del_nodes:
            self.tree.pop(d, None)
                        
        assert all([len(k.contained_labels) > 0 or k.class_num != -1 for k in self.tree.values()])
        self.add_all_children()
        
    def load_imagenet_info(self, ds_path, ds_info_path):
        files = os.listdir(os.path.join(ds_path, 'train'))
        in_wnids = [f for f in files if f[0]=='n'] 

        f = open(os.path.join(ds_info_path, 'words.txt'))
        wnid_to_name = [l.strip() for l in f.readlines()]
        wnid_to_name = {l.split('\t')[0]: l.split('\t')[1] \
                             for l in wnid_to_name}

        with open(os.path.join(ds_info_path, 'imagenet_class_index.json'), 'r') as f:
            base_map = json.load(f)
            wnid_to_num = {v[0]: int(k) for k, v in base_map.items()}
            num_to_name = {int(k): v[1] for k, v in base_map.items()}

        return in_wnids, wnid_to_name, wnid_to_num, num_to_name
    
    def get_node(self, wnid):
        if wnid not in self.tree:
            self.tree[wnid] = Node(wnid, name=self.wnid_to_name[wnid])
        return self.tree[wnid]
    
    def get_tree(self):
        return self.tree
    
    def get_class_name(self, wnid):
        return self.tree[wnid].get_name()
    
    # caculate simple edge distance
    def calculate_distance(self, wnid1, wnid2, mode='sum', gamma=1):
        if mode not in ['sum','harmonic','min','max']:
            assert("Please choose mode in 'sum','harmonic' and 'min'")
        min_dis=-1
        parents_trees1=self.generate_parents_tree(wnid1)
        parents_trees2=self.generate_parents_tree(wnid2)
        def d(start,end):
            distance=0
            if start>end:
                assert('Please check the input numbers')
            for i in range(start,end+1,1):
                distance+=pow(gamma,i)
            return distance
        #find the index where parents are different
        for t1 in parents_trees1:
            for t2 in parents_trees2:
                l1,l2=len(t1),len(t2)
                min_len = min(l1,l2)
                for i in range(min_len):
                    if t1[l1-1-i]!=t2[l2-1-i]:
                        break
                #calculate the distance (the index is tried)
                if mode == 'sum':  
                    dis = d(i-1,l2-2)+d(i-1,l1-2)
                elif mode == 'harmonic':
                    dis = d(i-1,l2-2)*d(i-1,l1-2)/(d(i-1,l2-2)+d(i-1,l1-2))
                elif mode == 'min':
                    dis = min(d(i-1,l2-2),d(i-1,l1-2))
                elif mode == 'max':
                    dis = max(d(i-1,l2-2),d(i-1,l1-2))

                if min_dis ==-1 or dis<min_dis:
                    min_dis=dis
        return min_dis
    
    def calculate_wu_palmer_similarity(self, wnid1, wnid2):
        max_sim=-1
        parents_trees1=self.generate_parents_tree(wnid1)
        parents_trees2=self.generate_parents_tree(wnid2)
        #find the index where parents are different
        flag=0
        for t1 in parents_trees1:
            for t2 in parents_trees2:
                l1,l2=len(t1),len(t2)
                min_len = min(l1,l2)
                for i in range(min_len):
                    if t1[l1-1-i]!=t2[l2-1-i]:
                        flag=1
                        break

                sim = 2*(i-flag)/(l1+l2-2)
                if sim>max_sim:
                    max_sim = sim

        return max_sim
    
    def calculate_jcn_similarity(self, wnid1, wnid2):
        if wnid1 == wnid2:
            return 1
        similarity=-1
        parents_trees1=self.generate_parents_tree(wnid1)
        parents_trees2=self.generate_parents_tree(wnid2)
        for t1 in parents_trees1:
            for t2 in parents_trees2:
                l1,l2=len(t1),len(t2)
                min_len = min(l1,l2)
                for i in range(min_len):
                    if t1[l1-1-i]!=t2[l2-1-i]:
                        break
                #The Jiang and Conrath similarity, 1/(IC(c1)+IC(c2)-2*IC(LCS(c1,c2)))
                sim = 1/(math.log(len(self.tree)/(len(self.tree[wnid1].get_children_all())+1)) \
                         + math.log(len(self.tree)/(len(self.tree[wnid2].get_children_all())+1)) \
                         - 2*math.log(len(self.tree)/(len(self.tree[t1[l1-i]].get_children_all())+1)))
                if sim>similarity:
                    similarity = sim
                    
        return similarity
                
    #for each node, add all the contained labels
    def add_all_labels(self):
        for k,v in self.wnid_to_num.items():
            parents_list=set()
            #get the parent trees and generate a list for all the parents.
            parents_trees = self.generate_parents_tree(k)
            for t in parents_trees:
                #the first one is the node itself so it shouldn't be contained
                for i in range(len(t)):
                    parents_list.add(t[i])
            #add child
            for parent in parents_list:
                self.tree[parent].add_contained_labels(k,v)

    #for each node, add all the childrens
    def add_all_children(self):
        for k in self.tree.keys():
            parents_list=[]
            #get the parent trees and generate a list for all the parents.
            parents_trees = self.generate_parents_tree(k)
            for t in parents_trees:
                #the first one is the node itself so it shouldn't be contained
                for i in range(1,len(t)):
                    if t[i] not in parents_list:
                        parents_list.append(t[i])
            #add child
            for parent in parents_list:
                self.tree[parent].add_children_all(k)   
    
    #get a parent tree for calculating the edge similarity
    def generate_parents_tree(self, wnid):
        parents_tree=[[wnid]]
        #Loop to get parent trees
        while 1:
            flag=1
            for i in range(len(parents_tree)):
                #get the parent for the last node
                parents = self.tree[parents_tree[i][-1]].get_parents()
                #if there's parent for the node
                if len(parents)>0:
                    #delete the tree at first
                    tmp = parents_tree[i]
                    parents_tree.remove(tmp)
                    #add the trees which contain the parents of the last node
                    for p in parents:
                        tmp.append(p.get_wnid())
                        parents_tree.insert(i,tmp.copy())
                        tmp.remove(p.get_wnid())
                        #index setting
                        i+=1
                    i-=1
                    # mark there are still sone parents not be contained
                    flag = flag and False
            if flag:
                break
        return parents_tree