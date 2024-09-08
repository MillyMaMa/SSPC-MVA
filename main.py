import os
import numpy as np
import argparse
import os
from train import trainer
from train_real import trainer as trainer_real
from test_shapenet import Tester
from test_real import Tester as tester_real

def get_parser():
    parser = argparse.ArgumentParser(description='Point cloud Completion')

    parser.add_argument('--test', type = bool, default = False, choices = [True, False])
    parser.add_argument('--test_dir', type=str, default="", help='') 
    parser.add_argument('--weight', type=str,  help='', default="")  

    parser.add_argument('--experiment_id', type=str, default= 'tmp', help='experiment id ')  
    parser.add_argument('--dataset_name', type=str, default='ShapeNet', help='dataset name', choices = ['ShapeNet'])
    parser.add_argument('--root', type=str, default='/data/shapeNet/', help='dataset directory') 
    parser.add_argument('--class_name', type=str, default='car', help='class of dataset', choices = ['plane', 'car', 'chair']) 
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)') 
    
    parser.add_argument('--snapshot_dir', type=str, default="", help='') 
    parser.add_argument('--output_dir', type=str, default="", help='') 
    parser.add_argument('--lr', type=float, default=0.001)  
    
    parser.add_argument('--epochs', type=int, default=3000, metavar='N', help='Number of episode to train ')
    parser.add_argument('--resume', type=str, default=None, metavar='N', help='checkpoint address')

    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N', help='Save snapshot interval ') 

    parser.add_argument('--exp_name', type=str, default="default", metavar='N', help='Name of the experiment')
    parser.add_argument('--k', type=int, default=None, metavar='N', help='Num of nearest neighbors to use for KNN')                
    parser.add_argument('--pretrain', type = bool, default = False, choices = [True, False])                  
    parser.add_argument('--num_points', type=int, default=8192, metavar='N', help='Num_points before removing (Original num_points)') 
    parser.add_argument('--remov_ratio', type=int, default=8, metavar='N', help='How much part of point cloud is goint to be removed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()

    if args.test is False: 
        output_dir = os.path.join(args.snapshot_dir, args.experiment_id, "outputs" )
        args.output_dir = output_dir

        isdir = os.path.isdir(output_dir)

        if isdir is False:         
            os.makedirs(args.output_dir)
            os.makedirs(args.output_dir+'/train')
            os.makedirs(args.output_dir+'/test/gt')
            os.makedirs(args.output_dir+'/test/ipc/0')
            os.makedirs(args.output_dir+'/test/ipc/1')
            os.makedirs(args.output_dir+'/test/ipc/2')
            os.makedirs(args.output_dir+'/test/ipc/3')
            os.makedirs(args.output_dir+'/test/ipc/4')
            
            os.makedirs(args.output_dir+'/test/cpc/0')
            os.makedirs(args.output_dir+'/test/cpc/1')
            os.makedirs(args.output_dir+'/test/cpc/2')
            os.makedirs(args.output_dir+'/test/cpc/3')
            os.makedirs(args.output_dir+'/test/cpc/4')

            os.makedirs(args.output_dir+'/SummaryWriter/train')
            os.makedirs(args.output_dir+'/SummaryWriter/val')

        np.save(args.output_dir+'/result.npy',100*np.ones(4))

    else:  
        args.output_dir = args.test_dir + '%s' % args.experiment_id
        isdir = os.path.isdir(args.output_dir)
        if isdir is False:         
            if (args.dataset_name == 'ShapeNet'): 
                os.makedirs(args.output_dir)
                os.makedirs(args.output_dir+'/gt')
                os.makedirs(args.output_dir+'/ipc/0')
                os.makedirs(args.output_dir+'/ipc/1')
                os.makedirs(args.output_dir+'/ipc/2')
                os.makedirs(args.output_dir+'/ipc/3')
                os.makedirs(args.output_dir+'/ipc/4')
                
                os.makedirs(args.output_dir+'/cpc/0')
                os.makedirs(args.output_dir+'/cpc/1')
                os.makedirs(args.output_dir+'/cpc/2')
                os.makedirs(args.output_dir+'/cpc/3')
                os.makedirs(args.output_dir+'/cpc/4')
            elif (args.dataset_name == 'MatterPort') or (args.dataset_name == 'ScanNet') or (args.dataset_name == 'KITTI'):
                os.makedirs(args.output_dir)
                os.makedirs(args.output_dir+'/cpc')
                os.makedirs(args.output_dir+'/ipc')
                os.makedirs(args.output_dir+'/v')
 

    if args.test is False:
        if (args.dataset_name == 'ShapeNet'): 
            train = trainer(args) 
            train.run(args)

        elif (args.dataset_name == 'MatterPort') or (args.dataset_name == 'ScanNet') or (args.dataset_name == 'KITTI'):
            train = trainer_real(args)
            train.run(args)

    else:  
        if os.path.isdir(args.test_dir) is False:
            os.makedirs(args.test_dir)

        if (args.dataset_name == 'ShapeNet'): 
            test = Tester(args)
            test.run()
        elif (args.dataset_name == 'MatterPort') or (args.dataset_name == 'ScanNet') or (args.dataset_name == 'KITTI'):
            test = tester_real(args)
            test.run()   


   