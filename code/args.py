import argparse

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type = str, default = '../data/2007_1_1/',
				   help='path to data')
	parser.add_argument('--model_path', type = str, default = '../model_save/',
				   help='path to save model')
	parser.add_argument('--L_n', type = int, default = 22309,
				   help = 'number of LP node')
	parser.add_argument('--F_n', type = int, default = 6498,
				   help = 'number of fund node')
	parser.add_argument('--I_n', type = int, default = 90024,
				   help = 'number of investor node')
	parser.add_argument('--C_n', type = int, default = 45906,
				   help = 'number of company node')
	parser.add_argument('--in_f_d', type = int, default = 384,
				   help = 'input feature dimension')
	parser.add_argument('--embed_d', type = int, default = 384,
				   help = 'embedding dimension')
	parser.add_argument('--lr', type = int, default = 0.001,
				   help = 'learning rate')
	parser.add_argument('--batch_s', type = int, default = 10000,
				   help = 'batch size')
	parser.add_argument('--mini_batch_s', type = int, default = 200,
				   help = 'mini batch size')
	parser.add_argument('--train_iter_n', type = int, default = 50,
				   help = 'max number of training iteration')
	parser.add_argument('--walk_n', type = int, default = 5,
				   help='number of walk per root node')
	parser.add_argument('--walk_L', type = int, default = 15,
				   help='length of each walk')
	parser.add_argument('--window', type = int, default = 5,
				   help='window size for relation extration')
	parser.add_argument("--random_seed", default = 10, type = int)
	parser.add_argument('--train_test_label', type= int, default = 0,
				   help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
	parser.add_argument('--save_model_freq', type = float, default = 2,
				   help = 'number of iterations to save model')
	parser.add_argument("--cuda", default = 0, type = int)
	parser.add_argument("--checkpoint", default = '', type=str)

	args = parser.parse_args()

	return args
