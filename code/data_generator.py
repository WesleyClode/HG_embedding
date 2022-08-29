import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *


class input_data(object):
	def __init__(self, args):
		self.args = args
		# self.node_type_list = ['l','f','i','c']
		# self.node_number_list = [self.args.L_n, self.args.F_n, self.args.I_n, self.args.C_n]

		# 构建节点的一阶邻居
		l_f_list_train = [[] for k in range(self.args.L_n)]
		f_l_list_train = [[] for k in range(self.args.F_n)]
		f_i_list_train = [[] for k in range(self.args.F_n)] 
		i_f_list_train = [[] for k in range(self.args.I_n)] 
		i_c_list_train = [[] for k in range(self.args.I_n)]
		c_i_list_train = [[] for k in range(self.args.C_n)]
		c_c_list_train = [[] for k in range(self.args.C_n)]

		relation_f = ["l_f_list_train.txt", "f_l_list_train.txt",\
		 "f_i_list_train.txt", "i_f_list_train.txt",\
		 "i_c_list_train.txt","c_i_list_train.txt",\
		 "c_c_list_train.txt"]

		#store relational data 
		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(self.args.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				for j in range(len(neigh_list_id)):
					rela_list_train = eval(f_name[:-4])
					# rela_list_train[node_id].append(f_name[2]+str(neigh_list_id[j]))
					list_train_set = set(rela_list_train[node_id])
					list_train_set.add(f_name[2]+str(neigh_list_id[j]))
					rela_list_train[node_id] = list(list_train_set)
			neigh_f.close()

		l_neigh_list_train = [[] for k in range(self.args.L_n)]
		for i in range(self.args.L_n):
			l_neigh_list_train[i] += l_f_list_train[i]
    
		f_neigh_list_train = [[] for k in range(self.args.F_n)]
		for i in range(self.args.F_n):
			f_neigh_list_train[i] += f_l_list_train[i]
			f_neigh_list_train[i] += f_i_list_train[i] 
    
		i_neigh_list_train = [[] for k in range(self.args.I_n)]
		for i in range(self.args.I_n):
			i_neigh_list_train[i] += i_f_list_train[i]
			i_neigh_list_train[i] += i_c_list_train[i] 
    
		c_neigh_list_train = [[] for k in range(self.args.C_n)]
		for i in range(self.args.C_n):
			c_neigh_list_train[i] += c_i_list_train[i]
			c_neigh_list_train[i] += c_c_list_train[i]             
            
		self.l_f_list_train =  l_f_list_train
		self.f_l_list_train =  f_l_list_train
		self.f_i_list_train = f_i_list_train
		self.i_f_list_train = i_f_list_train
		self.i_c_list_train = i_c_list_train
		self.c_i_list_train = c_i_list_train
		self.c_c_list_train = c_c_list_train   
		self.l_neigh_list_train = l_neigh_list_train    
		self.f_neigh_list_train = f_neigh_list_train    
		self.i_neigh_list_train = i_neigh_list_train    
		self.c_neigh_list_train = c_neigh_list_train    
            
		if self.args.train_test_label != 2:
			self.triple_sample_p = self.compute_sample_p()

		#store content pre-trained embedding
			def generate_content_embed(embed, embed_dir):
				embed_f = open(self.args.data_path + embed_dir, "r")
				for line in islice(embed_f, 1, None):
					values = line.split()
					index = int(values[0])
					embeds = np.asarray(values[1:], dtype='float32')
					embed[index] = embeds
				embed_f.close()
				return embed

			l_basic_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			self.l_basic_embed = generate_content_embed(l_basic_embed, "l_embedding.txt")
			
			f_basic_embed = np.zeros((self.args.F_n, self.args.in_f_d))
			self.f_basic_embed = generate_content_embed(f_basic_embed, "f_embedding.txt")
			
			i_basic_embed = np.zeros((self.args.I_n, self.args.in_f_d))
			self.i_basic_embed = generate_content_embed(i_basic_embed, "i_embedding.txt")
            
			c_basic_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			self.c_basic_embed = generate_content_embed(c_basic_embed, "c_embedding.txt")

			#store pre-trained network/content embedding
			l_net_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			f_net_embed = np.zeros((self.args.F_n, self.args.in_f_d))
			i_net_embed = np.zeros((self.args.I_n, self.args.in_f_d)) 
			c_net_embed = np.zeros((self.args.C_n, self.args.in_f_d)) 

			idx_var_dict = {'l':l_net_embed,'f':f_net_embed,'i':i_net_embed,'c':c_net_embed}
			
			net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				node = re.split(' ', line)
				if len(node) and (node[0] in idx_var_dict.keys()):
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					idx_var_dict[node[0]][int(node[1:])] = embeds
			net_e_f.close()

			self.l_net_embed = l_net_embed
			self.f_net_embed = f_net_embed
			self.i_net_embed = i_net_embed
			self.c_net_embed = c_net_embed

			def generate_rela_net_embed(rela_net_embed, latter_net_embed, rela_list_train, node_n):
				for i in range(node_n):
					if len(rela_list_train[i]):
						for j in range(len(rela_list_train[i])):
							l_id = int(rela_list_train[i][j][1:])
							rela_net_embed[i] = np.add(rela_net_embed[i], latter_net_embed[l_id])
						rela_net_embed[i] = rela_net_embed[i] / len(rela_list_train[i])
				return rela_net_embed

			f_l_net_embed = np.zeros((self.args.F_n, self.args.in_f_d))
			f_l_net_embed = generate_rela_net_embed(f_l_net_embed, l_net_embed, f_l_list_train, self.args.F_n)
			
			f_i_net_embed = np.zeros((self.args.F_n, self.args.in_f_d))
			f_i_net_embed = generate_rela_net_embed(f_i_net_embed, i_net_embed, f_i_list_train, self.args.F_n)
			
			i_f_net_embed = np.zeros((self.args.I_n, self.args.in_f_d))
			i_f_net_embed = generate_rela_net_embed(i_f_net_embed, f_net_embed, i_f_list_train, self.args.I_n)

			i_c_net_embed = np.zeros((self.args.I_n, self.args.in_f_d))
			i_c_net_embed = generate_rela_net_embed(i_c_net_embed, c_net_embed, i_c_list_train, self.args.I_n)

			c_i_net_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			c_i_net_embed = generate_rela_net_embed(c_i_net_embed, i_net_embed, c_i_list_train, self.args.C_n)

			c_c_net_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			c_c_net_embed = generate_rela_net_embed(c_c_net_embed, c_net_embed, c_c_list_train, self.args.C_n)

			self.f_l_net_embed = f_l_net_embed
			self.f_i_net_embed = f_i_net_embed
			self.i_f_net_embed = i_f_net_embed
			self.i_c_net_embed = i_c_net_embed
			self.c_i_net_embed = c_i_net_embed
			self.c_c_net_embed = c_c_net_embed

			#store neighbor set from random walk sequence 
			l_neigh_list_train = [[[] for i in range(self.args.L_n)] for j in range(4)]
			f_neigh_list_train = [[[] for i in range(self.args.F_n)] for j in range(4)]
			i_neigh_list_train = [[[] for i in range(self.args.I_n)] for j in range(4)]
			c_neigh_list_train = [[[] for i in range(self.args.C_n)] for j in range(4)]	

			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			node_type_list = ['l','f','i','c']
			
			def generate_neigh_list_train(node_id, neigh_list, neigh_list_train):
				if len(node_id) > 1:
					for j in range(len(neigh_list)):
						neigh_idx = node_type_list.index(neigh_list[j][0])
						neigh_number = int(neigh_list[j][1:])
						node_number = int(node_id[1:])
						neigh_list_train[neigh_idx][node_number].append(neigh_number)

				return neigh_list_train
			

			for line in het_neigh_train_f:
				line = line.strip()
				node_id = re.split(':', line)[0]
				neigh = re.split(':', line)[1]
				neigh_list = re.split(',', neigh)

				if node_id[0] == 'l':
					l_neigh_list_train = generate_neigh_list_train(node_id, neigh_list, l_neigh_list_train)
				if node_id[0] == 'f':
					f_neigh_list_train = generate_neigh_list_train(node_id, neigh_list, f_neigh_list_train)
				if node_id[0] == 'i':
					i_neigh_list_train = generate_neigh_list_train(node_id, neigh_list, i_neigh_list_train)
				if node_id[0] == 'c':
					c_neigh_list_train = generate_neigh_list_train(node_id, neigh_list, c_neigh_list_train)

			het_neigh_train_f.close()
			#print a_neigh_list_train[0][1]

			#store top neighbor set (based on frequency) from random walk sequence 
			l_neigh_list_train_top = [[[] for i in range(self.args.L_n)] for j in range(4)]
			f_neigh_list_train_top = [[[] for i in range(self.args.F_n)] for j in range(4)]
			i_neigh_list_train_top = [[[] for i in range(self.args.I_n)] for j in range(4)]
			c_neigh_list_train_top = [[[] for i in range(self.args.C_n)] for j in range(4)]
			top_k = [10, 10, 10, 10] #fix each neighor type size 
			
			def generate_top_neigh_set(node_n, neigh_list_train, neigh_list_train_top):
				for i in range(node_n):
					for j in range(len(top_k)):
						neigh_list_train_temp = Counter(neigh_list_train[j][i])
						top_list = neigh_list_train_temp.most_common(top_k[j])
						neigh_size = top_k[j]
						for k in range(len(top_list)):
							neigh_list_train_top[j][i].append(int(top_list[k][0]))
						if len(neigh_list_train_top[j][i]) and len(neigh_list_train_top[j][i]) < neigh_size:
							for l in range(len(neigh_list_train_top[j][i]), neigh_size):
								neigh_list_train_top[j][i].append(random.choice(neigh_list_train_top[j][i]))
				return neigh_list_train_top

			l_neigh_list_train_top = generate_top_neigh_set(self.args.L_n, l_neigh_list_train, l_neigh_list_train_top)
			f_neigh_list_train_top = generate_top_neigh_set(self.args.F_n, f_neigh_list_train, f_neigh_list_train_top)
			i_neigh_list_train_top = generate_top_neigh_set(self.args.I_n, i_neigh_list_train, i_neigh_list_train_top)
			c_neigh_list_train_top = generate_top_neigh_set(self.args.C_n, c_neigh_list_train, c_neigh_list_train_top)
			
			l_neigh_list_train[:] = []
			f_neigh_list_train[:] = []
			i_neigh_list_train[:] = []
			c_neigh_list_train[:] = []

			self.l_neigh_list_train = l_neigh_list_train_top
			self.f_neigh_list_train = f_neigh_list_train_top
			self.i_neigh_list_train = i_neigh_list_train_top
			self.c_neigh_list_train = c_neigh_list_train_top

			#store ids of lfic used in training 
			train_id_list = [[] for i in range(4)]
			def generate_train_id_list(idx, node_n, neigh_list_train_top, train_id_list):
				for l in range(node_n):
					if len(neigh_list_train_top[idx][l]):
						train_id_list[idx].append(l)
				return np.array(train_id_list[idx])
			
			self.l_train_id_list = generate_train_id_list(0, self.args.L_n, l_neigh_list_train_top, train_id_list)
			self.f_train_id_list = generate_train_id_list(1, self.args.F_n, f_neigh_list_train_top, train_id_list)
			self.i_train_id_list = generate_train_id_list(2, self.args.I_n, i_neigh_list_train_top, train_id_list)
			self.c_train_id_list = generate_train_id_list(3, self.args.C_n, c_neigh_list_train_top, train_id_list)

			#print (len(self.v_train_id_list))		


	def het_walk_restart(self):
		l_neigh_list_train = [[] for k in range(self.args.L_n)]
		f_neigh_list_train = [[] for k in range(self.args.F_n)]
		i_neigh_list_train = [[] for k in range(self.args.I_n)]
		c_neigh_list_train = [[] for k in range(self.args.C_n)]
		#generate neighbor set via random walk with restart
		node_n = [self.args.L_n, self.args.F_n, self.args.I_n, self.args.C_n]
		node_type = ['l','f','i','c']
		neigh_train_list = [l_neigh_list_train, f_neigh_list_train, i_neigh_list_train, c_neigh_list_train]
		self_neigh_train_list = [self.l_neigh_list_train, self.f_neigh_list_train, self.i_neigh_list_train, self.c_neigh_list_train]
		list_train_list = [self.l_f_list_train, self.f_i_list_train, self.i_c_list_train, self.c_c_list_train]
		neighbor_size = 100
		all_nodes = self.args.L_n + self.args.F_n + self.args.I_n + self.args.C_n
		l_constaint = neighbor_size*1.1*(self.args.L_n/all_nodes)
		f_constaint = neighbor_size*1.1*(self.args.F_n*2/all_nodes)
		i_constaint = neighbor_size*1.1*(self.args.I_n/all_nodes)
		c_constaint = neighbor_size*1.1*(self.args.C_n/all_nodes)
		constaint_list = [l_constaint, f_constaint, i_constaint, c_constaint]
		for i in range(len(node_n)):
			for j in range(node_n[i]):
				neigh_temp = list_train_list[i][j]
				neigh_train = neigh_train_list[i][j]
				curNode = node_type[i] + str(j)
				if len(neigh_temp):
					neigh_L = 0
					l_L = 0
					f_L = 0
					i_L = 0
					c_L = 0
					node_L_list = [l_L,f_L,i_L,c_L]
					count = 0
					while (neigh_L < neighbor_size) & (count<=300): #maximum neighbor size = 100
						rand_p = random.random() #return p
						if rand_p > 0.5:
							curNode_idx = node_type.index(curNode[0])
							# new curnode
							curNode = random.choice(self_neigh_train_list[curNode_idx][int(curNode[1:])])
							if curNode in neigh_train:
								count += 1
							curNode_idx = node_type.index(curNode[0])
							if node_L_list[curNode_idx] < constaint_list[curNode_idx]: #size constraint (make sure each type of neighobr is sampled)
								neigh_train.append(curNode)
								neigh_L += 1
								node_L_list[curNode_idx] += 1
						else:
							curNode = node_type[i] + str(j)

		for i in range(4):
			for j in range(node_n[i]):
				neigh_train_list[i][j] = list(neigh_train_list[i][j])

		neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
		for i in range(4):
			for j in range(node_n[i]):
				neigh_train = neigh_train_list[i][j]
				curNode = node_type[i] + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()


	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L

		total_triple_n = [0.0] * 16 # 16 kinds of triples
		het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''

		node_kind_list = ['l','f','i','c']
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					for k in range(j - window, j + window + 1):
						if k and k < walk_L and k != j:
							neighNode = path[k]
							tri_idx = node_kind_list.index(centerNode[0])*4 + node_kind_list.index(neighNode[0])
							total_triple_n[tri_idx] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
		print("sampling ratio computing finish.")

		return total_triple_n


	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(16)]
		window = self.args.window
		walk_L = self.args.walk_L
		L_n = self.args.L_n # number of nodes
		F_n = self.args.F_n
		I_n = self.args.I_n
		C_n = self.args.C_n
		triple_sample_p = self.triple_sample_p # use sampling to avoid memory explosion

		het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''
		node_kind_list = ['l','f','i','c']
		node_number_list = [L_n,F_n,I_n,C_n]
		list_train_list = [self.l_f_list_train, self.f_i_list_train, self.i_c_list_train, self.c_i_list_train]
		
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					for k in range(j - window, j + window + 1):
						if k and k < walk_L and k != j:
							neighNode = path[k]
							center_idx = node_kind_list.index(centerNode[0])
							neigh_idx = node_kind_list.index(neighNode[0])	
							tri_idx = center_idx*4 + neigh_idx
							if random.random() < triple_sample_p[tri_idx]:
								node_number = node_number_list[neigh_idx]
								negNode = random.randint(0, node_number - 1)
								while len(list_train_list[neigh_idx][negNode]) == 0:
									negNode = random.randint(0, node_number_list[neigh_idx] - 1)
								# random negative sampling get similar performance as noise distribution sampling
								triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
								triple_list[tri_idx].append(triple)
		het_walk_f.close()

		return triple_list




