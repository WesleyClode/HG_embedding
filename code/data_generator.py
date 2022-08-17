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
					eval(f_name[:-4])[node_id].append(f_name[0]+str(neigh_list_id[j]))
			neigh_f.close()


		#investor neighbor: fund + company
		i_neigh_list_train = [[] for k in range(self.args.I_n)]
		for i in range(self.args.I_n):
			i_neigh_list_train[i] += i_f_list_train[i]
			i_neigh_list_train[i] += i_c_list_train[i] 

		self.i_f_list_train =  i_f_list_train
		self.i_c_list_train = i_c_list_train
		self.i_neigh_list_train = i_neigh_list_train

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
			
			c_basic_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			self.c_basic_embed = generate_content_embed(c_basic_embed, "c_basic_embed.txt")
			c_business_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			self.c_business_embed = generate_content_embed(c_business_embed, "c_business_embed.txt")
			c_financing_embed = np.zeros((self.args.C_n, self.args.in_f_d))
			self.c_financing_embed = generate_content_embed(c_financing_embed, "c_financing_embed.txt")
			l_basic_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			self.l_basic_embed = generate_content_embed(l_basic_embed, "l_basic_embed.txt")
			l_prefer_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			self.l_prefer_embed = generate_content_embed(l_prefer_embed, "l_prefer_embed.txt")
			l_combination_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			self.l_combination_embed = generate_content_embed(l_combination_embed, "l_combination_embed.txt")
			# l_basic_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			# self.l_basic_embed = generate_content_embed(l_basic_embed, "l_basic_embed.txt")
			# l_prefer_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			# self.l_prefer_embed = generate_content_embed(l_prefer_embed, "l_prefer_embed.txt")
			# l_combination_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			# self.l_combination_embed = generate_content_embed(l_combination_embed, "l_combination_embed.txt")
			# l_basic_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			# self.l_basic_embed = generate_content_embed(l_basic_embed, "l_basic_embed.txt")
			# l_prefer_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			# self.l_prefer_embed = generate_content_embed(l_prefer_embed, "l_prefer_embed.txt")
			# l_combination_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			# self.l_combination_embed = generate_content_embed(l_combination_embed, "l_combination_embed.txt")

			#store pre-trained network/content embedding
			l_net_embed = np.zeros((self.args.L_n, self.args.in_f_d))
			f_net_embed = np.zeros((self.args.F_n, self.args.in_f_d))
			i_net_embed = np.zeros((self.args.I_n, self.args.in_f_d)) 
			c_net_embed = np.zeros((self.args.C_n, self.args.in_f_d)) 
			idx_var_dict = {'l':l_net_embed,'f':f_net_embed,'i':i_net_embed,'c':c_net_embed}
			net_e_f = open(self.args.data_path + "node_net_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				index = re.split(' ', line)[0]
				if len(index) and (index[0] in idx_var_dict.keys()):
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					idx_var_dict[index[0]][int(index[1:])] = embeds
			net_e_f.close()

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
			l_neigh_list_train = [[[] for i in range(self.args.L_n)] for j in range(3)]
			f_neigh_list_train = [[[] for i in range(self.args.F_n)] for j in range(3)]
			i_neigh_list_train = [[[] for i in range(self.args.I_n)] for j in range(3)]
			c_neigh_list_train = [[[] for i in range(self.args.C_n)] for j in range(3)]	

			het_neigh_train_f = open(self.args.data_path + "het_neigh_train.txt", "r")
			
			
			def generate_neigh_list_train(node_id, neigh_list, neigh_list_train):
				if len(node_id) > 1:
					for j in range(len(neigh_list)):
						if neigh_list[j][0] == 'a':
							neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'p':
							neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
						elif neigh_list[j][0] == 'v':
							neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
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
			l_neigh_list_train_top = [[[] for i in range(self.args.L_n)] for j in range(3)]
			f_neigh_list_train_top = [[[] for i in range(self.args.F_n)] for j in range(3)]
			i_neigh_list_train_top = [[[] for i in range(self.args.C_n)] for j in range(3)]
			c_neigh_list_train_top = [[[] for i in range(self.args.V_n)] for j in range(3)]
			top_k = [10, 10, 10, 10] #fix each neighor type size 
			
			def generate_top_neigh_set(node_n, neigh_list_train, neigh_list_train_top):
				for i in range(node_n):
					for j in range(3):
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
			i_neigh_list_train_top = generate_top_neigh_set(self.args.C_n, i_neigh_list_train, i_neigh_list_train_top)
			c_neigh_list_train_top = generate_top_neigh_set(self.args.V_n, c_neigh_list_train, c_neigh_list_train_top)
			
			l_neigh_list_train[:] = []
			f_neigh_list_train[:] = []
			i_neigh_list_train[:] = []
			c_neigh_list_train[:] = []

			self.l_neigh_list_train = l_neigh_list_train_top
			self.f_neigh_list_train = f_neigh_list_train_top
			self.i_neigh_list_train = i_neigh_list_train_top
			self.c_neigh_list_train = c_neigh_list_train_top

			#store ids of author/paper/venue used in training 
			train_id_list = [[] for i in range(3)]
			for i in range(3):
				if i == 0:
					for l in range(self.args.A_n):
						if len(a_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.a_train_id_list = np.array(train_id_list[i])
				elif i == 1:
					for l in range(self.args.P_n):
						if len(p_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.p_train_id_list = np.array(train_id_list[i])
				else:
					for l in range(self.args.V_n):
						if len(v_neigh_list_train_top[i][l]):
							train_id_list[i].append(l)
					self.v_train_id_list = np.array(train_id_list[i])
			#print (len(self.v_train_id_list))		


	def het_walk_restart(self):
		a_neigh_list_train = [[] for k in range(self.args.A_n)]
		p_neigh_list_train = [[] for k in range(self.args.P_n)]
		v_neigh_list_train = [[] for k in range(self.args.V_n)]

		#generate neighbor set via random walk with restart
		node_n = [self.args.A_n, self.args.P_n, self.args.V_n]
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_temp = self.a_p_list_train[j]
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_temp = self.p_a_list_train[j]
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				else:
					neigh_temp = self.v_p_list_train[j]
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
				if len(neigh_temp):
					neigh_L = 0
					a_L = 0
					p_L = 0
					v_L = 0
					while neigh_L < 100: #maximum neighbor size = 100
						rand_p = random.random() #return p
						if rand_p > 0.5:
							if curNode[0] == "a":
								curNode = random.choice(self.a_p_list_train[int(curNode[1:])])
								if p_L < 46: #size constraint (make sure each type of neighobr is sampled)
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
							elif curNode[0] == "p":
								curNode = random.choice(self.p_neigh_list_train[int(curNode[1:])])
								if curNode != ('a' + str(j)) and curNode[0] == 'a' and a_L < 46:
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'v':
									if v_L < 11:
										neigh_train.append(curNode)
										neigh_L += 1
										v_L += 1
							elif curNode[0] == "v":
								curNode = random.choice(self.v_p_list_train[int(curNode[1:])])
								if p_L < 46:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
						else:
							if i == 0:
								curNode = ('a' + str(j))
							elif i == 1:
								curNode = ('p' + str(j))
							else:
								curNode = ('v' + str(j))

		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					a_neigh_list_train[j] = list(a_neigh_list_train[j])
				elif i == 1:
					p_neigh_list_train[j] = list(p_neigh_list_train[j])
				else:
					v_neigh_list_train[j] = list(v_neigh_list_train[j])

		neigh_f = open(self.args.data_path + "het_neigh_train.txt", "w")
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = a_neigh_list_train[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_train = p_neigh_list_train[j]
					curNode = "p" + str(j)
				else:
					neigh_train = v_neigh_list_train[j]
					curNode = "v" + str(j)
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
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n

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
		L_n = self.args.L_n
		F_n = self.args.F_n
		I_n = self.args.I_n
		C_n = self.args.C_n
		triple_sample_p = self.triple_sample_p # use sampling to avoid memory explosion

		het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''
		node_kind_list = ['l','f','i','c']
		node_number_list = [L_n,F_n,I_n,C_n]
		rela_list = [self.l_f_list_train, self.f_l_list_train, ]
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]

								tri_idx = node_kind_list.index(centerNode[0])*4 + node_kind_list.index(neighNode[0])
								if random.random() < triple_sample_p[tri_idx]:
									number = node_number_list[node_kind_list.index(centerNode[0])]
									negNode = random.randint(0, number - 1)

									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									# random negative sampling get similar performance as noise distribution sampling
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
		het_walk_f.close()

		return triple_list




