import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test
from huseyin_functions import get_run_name, distribute_apples
import argparse
import wandb


# main code

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size to train on (default: 8)')
	parser.add_argument('-n','--notes',type=str, default = None , help = 'wandb run notes')
	parser.add_argument('-pn','--project_name',type=str, default = "nsubat" , help = 'wandb project name')
	parser.add_argument('-e','--n_epoch',type = int, default = 100,help = 'number of total epochs')
	# parser.add_argument('-ms','--ms',type = float, default = None, help = 'ratio of labeled source imagesper batch')
	parser.add_argument('-mss','--ms_list',type = float, nargs = "+", default = None, help = 'list of ratio of labeled source images')
	parser.add_argument('-Mss','--Ms_list',type = int, nargs = "+", default = None, help = 'list of total number of labeled source images')
	parser.add_argument('-Mt','--Mt',type = int, default = 25, help = 'ratio of labeled target images')
	# parser.add_argument('-mts','--mt_list',type = float, nargs = "+", default = None, help = 'list of ratio of labeled target images')
	# parser.add_argument('-dc_dim','--dc_dim',type = int, default = 100, help = 'dimension of the domain classifier network')
	# parser.add_argument('-log','--log_wandb',type=bool, default= True, help="whether to log to wandb or not")
	# parser.add_argument('-l','--layers',type=int, help="number of layers")
	parser.add_argument('-ll','--layers_list',type=int, nargs = "+", help="list of number of layers")
	# parser.add_argument('-rn','--run_name',type=str, help="name of the run")
	# parser.add_argument('-Mts','--Mt_list',type = int, nargs = "+", default = None, help = 'list of number of labeled target images per batch')
	parser.add_argument('-dcms','--dimchange_multipliers',type = float, nargs = "+", default = None, help = 'list of dimchange multipliers. common for conv and linear layers.')
	parser.add_argument('-beta','--beta',type = float,default = 0.5, help = 'balance parameter between classfication and domain losses. beta =1 means zero contribution from domain loss.')
	parser.add_argument('-ns','--ns',type = float, default = None, help ='the ratio that determines the dataset length used')
	parser.add_argument('-r','--repeats',type = int, default = 1, help ='how many repeats')
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
	n_epoch = args.n_epoch
	beta = args.beta 
	ns = args.ns
	Mt = args.Mt 


	for _ in range(args.repeats):
		for layer in args.layers_list:
			for dcm in args.dimchange_multipliers:
				for Ms in args.Ms_list:
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
							"config":{"Ms": Ms, "Mt": Mt, "layer":layer, "beta":beta, "dcm":dcm, "ns":ns, "bs":batch_size}
							}
					

					# Logging setup (You can replace it with your preferred logging method)
					wandb.init(**wandb_kwargs)
					wandb.run.log_code('.')


					# load data

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

					dataset_source = datasets.MNIST(
						root='dataset',
						train=True,
						transform=img_transform_source,
						download=True
					)

					dataloader_source = torch.utils.data.DataLoader(
						dataset=dataset_source,
						batch_size=batch_size,
						shuffle=True,
						num_workers=8,
						drop_last = True)

					train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

					dataset_target = GetLoader(
						data_root=os.path.join(target_image_root, 'mnist_m_train'),
						data_list=train_list,
						transform=img_transform_target
					)

					dataloader_target = torch.utils.data.DataLoader(
						dataset=dataset_target,
						batch_size=batch_size,
						shuffle=True,
						num_workers=8,
						drop_last = True)

					# load model
	
					my_net = CNNModel(layer,dcm)

					# setup optimizer

					optimizer = optim.Adam(my_net.parameters(), lr=lr)

					loss_class = torch.nn.NLLLoss(reduction = "sum")
					loss_domain = torch.nn.NLLLoss(reduction = "sum")

					if cuda:
						my_net = my_net.cuda()
						loss_class = loss_class.cuda()
						loss_domain = loss_domain.cuda()

					for p in my_net.parameters():
						p.requires_grad = True
					

					# training
					best_accu_t = 0.0
					for epoch in range(n_epoch):

						len_dataloader = int(ns*(min(len(dataloader_source), len(dataloader_target))))
						data_source_iter = iter(dataloader_source)
						data_target_iter = iter(dataloader_target)

						Msx_list = distribute_apples(Ms-1,len_dataloader-1)
						Msx_list.append(1)
						Mtx_list = distribute_apples(Mt-1,len_dataloader-1)
						Mtx_list.append(1)

						for i in range(len_dataloader):

							Msx = Msx_list[i] # required number of labeled source data for per batch to ensure total of Ms 
							Mtx = Mtx_list[i]

							p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
							alpha = 2. / (1. + np.exp(-10 * p)) - 1

							# training model using source data
							data_source = data_source_iter.next()
							s_img, s_label = data_source

							my_net.zero_grad()
							batch_size = len(s_label)

							domain_label = torch.zeros(batch_size).long()

							if cuda:
								s_img = s_img.cuda()
								s_label = s_label.cuda()
								domain_label = domain_label.cuda()


							class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
							err_s_label = loss_class(class_output[:Msx,:], s_label[:Msx])
							err_s_domain = loss_domain(domain_output, domain_label)

							# training model using target data
							data_target = data_target_iter.next()
							t_img, t_label = data_target

							batch_size = len(t_img)

							domain_label = torch.ones(batch_size).long()

							if cuda:
								t_img = t_img.cuda()
								t_label = t_label.cuda()
								domain_label = domain_label.cuda()

							class_output, domain_output = my_net(input_data=t_img, alpha=alpha)
							err_t_label = loss_class(class_output[:Mtx,:], t_label[:Mtx])
							err_t_domain = loss_domain(domain_output, domain_label)
							err =(1-beta)*(err_t_domain + err_s_domain) + beta*(err_s_label + err_t_label)
							err.backward()
							optimizer.step()

							# sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
							#       % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
							#          err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
							# sys.stdout.flush()
							torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))

						print('\n')
						print('Ms: %d | Mt: %d | dcm: %.2f | layer: %d | Epoch: %d/%d' % (Ms,Mt,dcm,layer,epoch,n_epoch))
						accu_s = test(source_dataset_name)
						print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
						accu_t = test(target_dataset_name)
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
					