import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
args = read_args()

def list_flatten(list_array):
	list_array = np.squeeze(list_array)
	list = []
	nozero = []
	for i in range(len(list_array)):
		if list_array[i] != []:
			nozero.append(i)
	
	print(nozero)

	for i in range(len(list_array)):
		print("i_list: ", list_array[i])
		if list_array[i] == []:
			list.append(list_array[choice(nozero)])
		else:
			list.append(list_array[i])
	return list

class HetAgg(nn.Module):
	def __init__(self, args, feature_list, l_neigh_list_train, f_neigh_list_train, i_neigh_list_train,\
		 c_neigh_list_train, l_train_id_list, f_train_id_list, i_train_id_list, c_train_id_list):
		super(HetAgg, self).__init__()
		embed_d = args.embed_d
		in_f_d = args.in_f_d
		self.args = args 
		self.L_n = args.L_n
		self.F_n = args.F_n
		self.I_n = args.I_n
		self.C_n = args.C_n
		self.feature_list = feature_list
		self.l_neigh_list_train = l_neigh_list_train
		self.f_neigh_list_train = f_neigh_list_train
		self.i_neigh_list_train = i_neigh_list_train
		self.c_neigh_list_train = c_neigh_list_train
		self.l_train_id_list = l_train_id_list
		self.f_train_id_list = f_train_id_list
		self.i_train_id_list = i_train_id_list
		self.c_train_id_list = c_train_id_list

		self.l_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.f_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.i_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.c_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		
		self.l_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.f_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.i_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
		self.c_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

		self.l_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.f_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.i_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
		self.c_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)

		self.softmax = nn.Softmax(dim = 1)
		self.act = nn.LeakyReLU()
		self.drop = nn.Dropout(p = 0.5)
		self.bn = nn.BatchNorm1d(embed_d)
		self.embed_d = embed_d


	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				#nn.init.normal_(m.weight.data)
				m.bias.data.fill_(0.1)


	def content_agg(self, node_type, id_batch, func_content_rnn): #heterogeneous content aggregation
		# print(id_batch)
		# print('----------------------------------')
		node_idx = self.args.node_type_list.index(node_type)
		feature_idx_range = range(self.args.feature_range[node_idx],self.args.feature_range[node_idx+1])
		embed_d = self.embed_d
		embed_batch = []
		idx_list = [n for n in feature_idx_range]
		for idx in idx_list:
			embed_batch.append(self.feature_list[idx][id_batch])
			# except:
			# 	print(idx,'\n',id_batch,'\n',self.feature_list[idx],'\n') 
			# 	break 

		concate_embed = torch.cat(embed_batch, 1).view(len(id_batch[0]), len(idx_list), embed_d)
		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = func_content_rnn(concate_embed)

		return torch.mean(all_state, 0)

	def node_neigh_agg(self, id_batch, node_type): #type based neighbor aggregation with rnn 
		embed_d = self.embed_d
		batch_s = int(len(id_batch[0]) / 10)

		if node_type == 'l':
			neigh_agg = self.content_agg(node_type, id_batch, self.l_content_rnn).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.l_neigh_rnn(neigh_agg)
		if node_type == 'f':
			neigh_agg = self.content_agg(node_type, id_batch, self.f_content_rnn).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.f_neigh_rnn(neigh_agg)
		if node_type == 'i':
			neigh_agg = self.content_agg(node_type, id_batch, self.i_content_rnn).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.i_neigh_rnn(neigh_agg)
		if node_type == 'c':
			neigh_agg = self.content_agg(node_type, id_batch, self.c_content_rnn).view(batch_s, 10, embed_d)
			neigh_agg = torch.transpose(neigh_agg, 0, 1)
			all_state, last_state  = self.c_neigh_rnn(neigh_agg)
		neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)
		
		return neigh_agg


	def node_het_agg(self, id_batch, node_type): #heterogeneous neighbor aggregation
		# print(id_batch)
		l_neigh_batch = [[0] * 10] * len(id_batch)
		f_neigh_batch = [[0] * 10] * len(id_batch)
		i_neigh_batch = [[0] * 10] * len(id_batch)
		c_neigh_batch = [[0] * 10] * len(id_batch)

		# def gene_neigh_batch(node_type):
		node_neigh_list_train_dict = {'l':self.l_neigh_list_train,'f':self.f_neigh_list_train,
			'i':self.i_neigh_list_train,'c':self.c_neigh_list_train}
		for i in range(len(id_batch)):
			l_neigh_batch[i] = node_neigh_list_train_dict[node_type][0][id_batch[i]]
			f_neigh_batch[i] = node_neigh_list_train_dict[node_type][1][id_batch[i]]
			i_neigh_batch[i] = node_neigh_list_train_dict[node_type][2][id_batch[i]]
			c_neigh_batch[i] = node_neigh_list_train_dict[node_type][3][id_batch[i]]
			# return l_neigh_batch, f_neigh_batch, i_neigh_batch, c_neigh_batch
		# l_neigh_batch, f_neigh_batch, i_neigh_batch, c_neigh_batch = gene_neigh_batch(node_type)

		l_neigh_batch = np.reshape(l_neigh_batch, (1, -1))
		l_agg_batch = self.node_neigh_agg(l_neigh_batch, 'l')
		f_neigh_batch = np.reshape(f_neigh_batch, (1, -1))
		f_agg_batch = self.node_neigh_agg(f_neigh_batch, 'f')
		i_neigh_batch = list_flatten(i_neigh_batch)
		i_neigh_batch = np.reshape(i_neigh_batch, (1, -1))
		i_agg_batch = self.node_neigh_agg(i_neigh_batch, 'i')
		c_neigh_batch = list_flatten(c_neigh_batch)
		c_neigh_batch = np.reshape(c_neigh_batch, (1, -1))
		c_agg_batch = self.node_neigh_agg(c_neigh_batch, 'c')

		#attention module
		id_batch = np.reshape(id_batch, (1, -1))
		if node_type == 'l':
			a_agg_batch = self.l_content_agg(id_batch)
		elif node_type == 'f':
			a_agg_batch = self.f_content_agg(id_batch)
		elif node_type == 'i':
			a_agg_batch = self.i_content_agg(id_batch)
		elif node_type == 'c':
			a_agg_batch = self.c_content_agg(id_batch)

		a_agg_batch_2 = torch.cat((a_agg_batch, a_agg_batch), 1).view(len(a_agg_batch), self.embed_d * 2)
		l_agg_batch_2 = torch.cat((a_agg_batch, l_agg_batch), 1).view(len(a_agg_batch), self.embed_d * 2)
		f_agg_batch_2 = torch.cat((a_agg_batch, f_agg_batch), 1).view(len(a_agg_batch), self.embed_d * 2)
		i_agg_batch_2 = torch.cat((a_agg_batch, i_agg_batch), 1).view(len(a_agg_batch), self.embed_d * 2)
		c_agg_batch_2 = torch.cat((a_agg_batch, c_agg_batch), 1).view(len(a_agg_batch), self.embed_d * 2)

		#compute weights
		concate_embed = torch.cat((a_agg_batch_2, l_agg_batch_2, f_agg_batch_2,\
		 i_agg_batch_2, c_agg_batch_2), 1).view(len(a_agg_batch), 4, self.embed_d * 2)
		if node_type == 'l':
			atten_w = self.act(torch.bmm(concate_embed, self.l_neigh_att.unsqueeze(0).expand(len(a_agg_batch),\
			 *self.l_neigh_att.size())))
		elif node_type == 'f':
			atten_w = self.act(torch.bmm(concate_embed, self.f_neigh_att.unsqueeze(0).expand(len(a_agg_batch),\
			 *self.f_neigh_att.size())))
		elif node_type == 'i':
			atten_w = self.act(torch.bmm(concate_embed, self.i_neigh_att.unsqueeze(0).expand(len(a_agg_batch),\
			 *self.i_neigh_att.size())))
		elif node_type == 'c':
			atten_w = self.act(torch.bmm(concate_embed, self.c_neigh_att.unsqueeze(0).expand(len(a_agg_batch),\
			 *self.c_neigh_att.size())))
		atten_w = self.softmax(atten_w).view(len(a_agg_batch), 1, 4)

		#weighted combination
		concate_embed = torch.cat((a_agg_batch, l_agg_batch, f_agg_batch,\
		 i_agg_batch, c_agg_batch), 1).view(len(a_agg_batch), 4, self.embed_d)
		weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(a_agg_batch), self.embed_d)

		return weight_agg_batch


	def het_agg(self, triple_index, v_id_batch, pos_id_batch, neg_id_batch):
		embed_d = self.embed_d
		# batch processing
		# nine cases for academic data (author, paper, venue)
		node_type_list = ['l','f','i','c']
		if triple_index < 16:
			v_agg = self.node_het_agg(v_id_batch, node_type_list[triple_index//4])
			p_agg = self.node_het_agg(pos_id_batch, node_type_list[triple_index%4+1])
			n_agg = self.node_het_agg(neg_id_batch, node_type_list[triple_index%4+1])

		elif triple_index == 16: #save learned node embedding
			embed_file = open(self.args.data_path + "node_embedding.txt", "w")
			save_batch_s = self.args.mini_batch_s

			i = 0
			for id_list in [self.l_train_id_list, self.f_train_id_list, self.i_train_id_list, self.c_train_id_list]:
				i += 1
				batch_number = int(len(id_list) / save_batch_s)
				for j in range(batch_number):
					id_batch = id_list[j * save_batch_s : (j + 1) * save_batch_s]
					out_temp = self.node_het_agg(id_batch, i)
					out_temp = out_temp.data.cpu().numpy()

					for k in range(len(id_batch)):
						index = id_batch[k]
						embed_file.write(node_type_list[i-1] + str(index) + " ")
						for l in range(embed_d - 1):
							embed_file.write(str(out_temp[k][l]) + " ")
						embed_file.write(str(out_temp[k][-1]) + "\n")

				id_batch = id_list[batch_number * save_batch_s : -1]
				out_temp = self.node_het_agg(id_batch, i) 
				out_temp = out_temp.data.cpu().numpy()
				
				for k in range(len(id_batch)):
					index = id_batch[k]
					embed_file.write(node_type_list[i-1] + str(index) + " ")
					for l in range(embed_d - 1):
						embed_file.write(str(out_temp[k][l]) + " ")
					embed_file.write(str(out_temp[k][-1]) + "\n")
			embed_file.close()
			return [], [], []

		return v_agg, p_agg, n_agg


	def aggregate_all(self, triple_list_batch, triple_index):
		v_id_batch = [x[0] for x in triple_list_batch]
		pos_id_batch = [x[1] for x in triple_list_batch]
		neg_id_batch = [x[2] for x in triple_list_batch]

		v_agg, pos_agg, neg_agg = self.het_agg(triple_index, v_id_batch, pos_id_batch, neg_id_batch)

		return v_agg, pos_agg, neg_agg


	def forward(self, triple_list_batch, triple_index):
		v_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
		return v_out, p_out, n_out


def cross_entropy_loss(v_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
	batch_size = v_embed_batch.shape[0] * v_embed_batch.shape[1]
	
	c_embed = v_embed_batch.view(batch_size, 1, embed_d)
	pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
	neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

	out_p = torch.bmm(c_embed, pos_embed)
	out_n = - torch.bmm(c_embed, neg_embed)

	sum_p = F.logsigmoid(out_p)
	sum_n = F.logsigmoid(out_n)
	loss_sum = - (sum_p + sum_n)

	#loss_sum = loss_sum.sum() / batch_size

	return loss_sum.mean()

