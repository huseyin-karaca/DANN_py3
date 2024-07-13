import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import RandomSampler, BatchSampler, DataLoader, random_split, Subset
import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test
from huseyin_functions import get_run_name, distribute_apples
import argparse
import wandb
import torch


os.environ["WANDB_API_KEY"] = "033f1f51ec386bb1ab8514e244d5acd2ce396356"

# main code

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size to train on (default: 8)')
	parser.add_argument('-n','--notes',type=str, default = None , help = 'wandb run notes')
	parser.add_argument('-pn','--project_name',type=str, default = "nsubat" , help = 'wandb project name')
	parser.add_argument('-e','--n_epoch',type = int, default = 100,help = 'number of total epochs')
	parser.add_argument('-nw','--num_workers',type = int, default = 8, help= 'num_workers parameters of the dataloader')
	# parser.add_argument('-ms','--ms',type = float, default = None, help = 'ratio of labeled source imagesper batch')
	parser.add_argument('-mss','--ms_list',type = float, nargs = "+", default = None, help = 'list of ratio of labeled source images')
	parser.add_argument('-Mss','--Ms_list',type = int, nargs = "+", default = None, help = 'list of total number of labeled source images')
	parser.add_argument('-Mt','--Mt',type = int, default = 25, help = 'ratio of labeled target images')
	parser.add_argument('-Mts','--Mt_list',type = int, nargs = "+", default = None, help = 'list of labeled target images')
	# parser.add_argument('-dc_dim','--dc_dim',type = int, default = 100, help = 'dimension of the domain classifier network')
	# parser.add_argument('-log','--log_wandb',type=bool, default= True, help="whether to log to wandb or not")
	# parser.add_argument('-l','--layers',type=int, help="number of layers")
	parser.add_argument('-ll','--layers_list',type=int, nargs = "+", help="list of number of layers")
	parser.add_argument('-NsNt','--NsNt',type=int, nargs = "+", help="list of number of total source and target images")
	# parser.add_argument('-rn','--run_name',type=str, help="name of the run")
	# parser.add_argument('-Mts','--Mt_list',type = int, nargs = "+", default = None, help = 'list of number of labeled target images per batch')
	parser.add_argument('-dcms','--dimchange_multipliers',type = float, nargs = "+", default = None, help = 'list of dimchange multipliers. common for conv and linear layers.')
	parser.add_argument('-gms','--gammas',type = float, nargs = "+", default = None, help = 'list of gamma weights between source and target class losses. alpha=1 means zero source class loss')
	parser.add_argument('-beta','--beta',type = float,default = 0.5, help = 'balance parameter between classfication and domain losses. beta =1 means zero contribution from domain loss.')
	parser.add_argument('-Ns','--Ns',type = int, default = None, help ='the total number of data used in source')
	parser.add_argument('-Nt','--Nt',type = int, default = None, help ='the total number of data used in target')
	parser.add_argument('-r','--repeats',type = int, default = 1, help ='how many repeats')
	parser.add_argument("-ae","--adaptive_epochs",action='store_true')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	source_dataset_name = 'MNIST'
	target_dataset_name = 'mnist_m'
	source_image_root = os.path.join('dataset', source_dataset_name)
	target_image_root = os.path.join('dataset', target_dataset_name)
	model_root = 'models'
	cuda = True
	cudnn.benchmark = True
	lr = 1e-3
	batch_size = args.batch_size
	image_size = 28
	
	beta = args.beta 

	
	gamma = 0.5

	for repeat in range(args.repeats):
		for Mt in args.Mt_list:
			for Ms in args.Ms_list:
				for layer in args.layers_list:
					n_epoch = args.n_epoch if not args.adaptive_epochs else int(50*layer)
					for dcm in args.dimchange_multipliers:
						for Ns in args.NsNt:
							Nt = Ns
							# Ms = int(ms*batch_size)
							manual_seed = random.randint(1, 10000)
							random.seed(manual_seed)
							torch.manual_seed(manual_seed)
					
	
	
							# logging - (wandb)
			
							# run_name = get_run_name() if not args.run_name else args.run_name
							# run_folder = '/home/huseyin/fungtion/dannpy_yeniden/DANN_py3/runs/' + run_name
							# os.makedirs(run_folder, exist_ok=True)
	
	
							wandb_kwargs = {# "dir": run_folder,
									# "name": run_name,
									"project": args.project_name,
									"notes": args.notes,
									# "id": run_name, #wandb_id_finder_from_folder(self.run_folder) if args.mode == 'resume' else wandb.util.generate_id(),
									#"resume": 'allow',
									#"allow_val_change": True,
									"config":{"Ms": Ms, 
											"Mt": Mt, 
											"layer":layer,  
											"dcm":dcm, 
											"Ns":Ns, 
											"Nt":Nt,
											"bs":batch_size,
											"repeat":repeat+1,
											"gamma":gamma}
									}
							
	
							# Logging setup (You can replace it with your preferred logging method)
							wandb.init(**wandb_kwargs)
							wandb.run.log_code('.')
	
	
							# load data
	
							## define transformations
							img_transform_source = transforms.Compose([
								transforms.Resize(image_size),
								transforms.ToTensor(),
								transforms.Normalize(mean=(0.1307,), std=(0.3081,))
							])
	
							img_transform_target = transforms.Compose([
								transforms.Resize(image_size),
								transforms.ToTensor(),
								transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
							])
	
							## read source and target datasets for the first time:
							dataset_source = datasets.MNIST(
								root='dataset',
								train=True,
								transform=img_transform_source,
								download=True
							)
	
							dataset_target = GetLoader(
								data_root=os.path.join(target_image_root, 'mnist_m_train'),
								data_list=os.path.join(target_image_root, 'mnist_m_train_labels.txt'),
								transform=img_transform_target
							)
	
							## create subsets of the original source and target datasets. choose random Ns and Nt indices respectively:
							dataset_source = Subset(dataset_source, indices=random.sample(range(len(dataset_source)),Ns))
							dataset_target = Subset(dataset_target, indices=random.sample(range(len(dataset_target)),Nt))
	
							## split labeled and unlabeled datasets:
							dataset_source_labeled, dataset_source_unlabeled = random_split(dataset_source, [Ms, Ns-Ms])
							dataset_target_labeled, dataset_target_unlabeled = random_split(dataset_target, [Mt, Nt-Mt])
	
	
							## create labeled and unlabeled dataloaders seperately:
							## (in order to prevent using more labeled total data than specified Ms and Mt values)
	
							### source dataloaders
							dataloader_source_labeled = DataLoader(
								dataset=dataset_source_labeled,
								batch_size=1,
								shuffle=True,
								num_workers=args.num_workers,
								drop_last = True)
								
							dataloader_source_unlabeled = DataLoader(
								dataset=dataset_source_unlabeled,
								batch_size=batch_size,
								shuffle=True,
								num_workers=args.num_workers,
								drop_last = True)
	
							### target dataloaders
							dataloader_target_labeled = DataLoader(
								dataset=dataset_target_labeled,
								batch_size=1,
								shuffle=True,
								num_workers=args.num_workers,
								drop_last = True)
								
							dataloader_target_unlabeled = DataLoader(
								dataset=dataset_target_unlabeled,
								batch_size=batch_size,
								shuffle=True,
								num_workers=args.num_workers,
								drop_last = True)
	
							# load model
			
							my_net = CNNModel(layer,dcm,"mchange")
	
							# setup optimizer
	
							optimizer = optim.Adam(my_net.parameters(), lr=lr)
	
							# loss_class = torch.nn.NLLLoss(reduction = "sum")
							# loss_domain = torch.nn.NLLLoss(reduction = "sum")
							loss_class = torch.nn.NLLLoss(reduction = "mean")
							loss_domain = torch.nn.NLLLoss(reduction = "mean")
	
							if cuda:
								my_net = my_net.cuda()
								loss_class = loss_class.cuda()
								loss_domain = loss_domain.cuda()
	
							for p in my_net.parameters():
								p.requires_grad = True
							
	
							# training
							best_accu_t = 0.0
							for epoch in range(n_epoch):
	
								# create iterators:
								source_unlabeled_iter = iter(dataloader_source_unlabeled)
								source_labeled_iter = iter(dataloader_source_labeled)
								target_unlabeled_iter = iter(dataloader_target_unlabeled)
								target_labeled_iter = iter(dataloader_target_labeled)
	
								Ks = len(source_unlabeled_iter)
								Kt = len(target_unlabeled_iter)
								K = min(Ks,Kt)
	
								### distribution of number of labeled samples per batch
								Msx_list = distribute_apples(Ms,Ks)
								Mtx_list = distribute_apples(Mt,Kt)										
	
								for i in range(K): 
									
									# required number of labeled data for i'th batch to ensure the total of Ms or Mt 
									Msx = Msx_list[i] 
									Mtx = Mtx_list[i]
	
									p = float(i + epoch * K) / n_epoch / K
									alpha = 2. / (1. + np.exp(-10 * p)) - 1
	
									my_net.zero_grad()
	
									# training model using source data
									
									source_unlabeled_img, _ = next(source_unlabeled_iter)
									if Msx:
										source_labeled_img, source_labeled_label  = next(source_labeled_iter)
										for i in range(Msx-1):
											source_labeled_img_temp, source_labeled_label_temp  = next(source_labeled_iter)
											source_labeled_img = torch.cat((source_labeled_img,source_labeled_img_temp),dim = 0)
											source_labeled_label = torch.cat((source_labeled_label,source_labeled_label_temp),dim = 0)
	
										s_img = torch.cat((source_unlabeled_img[:-Msx,:,:,:],source_labeled_img[:Msx,:,:,:]),dim = 0)
										s_label = source_labeled_label
									else:
										s_img = source_unlabeled_img
										s_label = torch.rand([0]).long()
									
									
	
									domain_label = torch.zeros(batch_size).long()
									
									# batch_size = len(s_label)
	
									if cuda:
										s_img = s_img.cuda()
										s_label = s_label.cuda()
										domain_label = domain_label.cuda()
	
	
									class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
									err_s_label = loss_class(class_output[(batch_size-Msx):,:], s_label)
									err_s_domain = loss_domain(domain_output, domain_label)
	
									# training model using target data
									target_unlabeled_img, _ = next(target_unlabeled_iter)
									if Mtx:
										target_labeled_img, target_labeled_label  = next(target_labeled_iter)
										for i in range(Mtx-1):
											target_labeled_img_temp, target_labeled_label_temp  = next(target_labeled_iter)
											target_labeled_img = torch.cat((target_labeled_img,target_labeled_img_temp),dim = 0)
											target_labeled_label = torch.cat((target_labeled_label,target_labeled_label_temp),dim = 0)
	
										t_img = torch.cat((target_unlabeled_img[:-Mtx,:,:,:],target_labeled_img[:Mtx,:,:,:]),dim = 0)
										t_label = target_labeled_label
									else:
										t_img = target_unlabeled_img
										t_label = torch.rand([0]).long()
	
									# batch_size = len(t_img)
	
									domain_label = torch.ones(batch_size).long()
	
									if cuda:
										t_img = t_img.cuda()
										t_label = t_label.cuda()
										domain_label = domain_label.cuda()
	
									class_output, domain_output = my_net(input_data=t_img, alpha=alpha)
									err_t_label = loss_class(class_output[(batch_size-Mtx):,:], t_label)
									err_t_domain = loss_domain(domain_output, domain_label)
	
									#err =(1-beta)*(err_t_domain + err_s_domain) + beta*(err_s_label + err_t_label)
									# err =(6000/NsNt)*(err_t_domain + err_s_domain) + (err_s_label + err_t_label) #summed loss
									# err = err_t_domain + err_s_domain +       err_s_label +            err_t_label  #average loss
									err = 0.5 * err_t_domain + 0.5 * err_s_domain + gamma*err_t_label + (1-gamma)* err_s_label  #average, gamma weighted loss
									err.backward()
									optimizer.step()
	
									# sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
									#       % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
									#          err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
									# sys.stdout.flush()
									torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))
	
								print('\n')
								print('Ms: %d | Mt: %d | dcm: %.2f | layer: %d | Epoch: %d/%d' % (Ms,Mt,dcm,layer,epoch+1,n_epoch))
								accu_s = test(source_dataset_name,args)
								print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
								accu_t = test(target_dataset_name,args)
								print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
	
								if accu_t > best_accu_t:
									best_accu_s = accu_s
									best_accu_t = accu_t
									torch.save(my_net, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))
	
								wandb.log({"err_s_label_train": err_s_label/Msx,
								"err_s_domain_train": err_s_domain/batch_size,
								"err_t_label_train": err_t_label/Mtx,
								"err_t_domain_train": err_t_domain/batch_size,
								"total_loss":err,
								"accu_s_test": accu_s,
								"accu_t_test": accu_t}, step=epoch)
	
							
							wandb.finish()
							
