import argparse
parser = argparse.ArgumentParser()
# training settings
parser.add_argument('--epoch', type=int, default=150, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

# device settings
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# dataset settings
parser.add_argument('--img_root', type=str, default='I:\\Stu-ZhongJunmin\\TrainDataset\\Imgs/')
parser.add_argument('--edge_root', type=str, default='I:\\Stu-ZhongJunmin\\TrainDataset\\Edge/')
parser.add_argument('--gt_root', type=str, default='I:\\Stu-ZhongJunmin\\TrainDataset\\GT/')
parser.add_argument('--test_img_root', type=str, default='I:\\Stu-ZhongJunmin\\TestDataset\\CAMO\\Imgs/')
parser.add_argument('--test_gt_root', type=str, default='I:\\Stu-ZhongJunmin\\TestDataset\\CAMO\\GT/')
parser.add_argument('--test_path', type=str, default=r'I:\Stu-ZhongJunmin\BAINet\save_path\Net_epoch_best.pth')
parser.add_argument('--test_img', type=str, default='I:\\Stu-ZhongJunmin\\TestDataset')
parser.add_argument('--test_save_path', type=str, default='./results')
parser.add_argument('--test_edge_path', type=str, default='./edge_results')
parser.add_argument('--test_init_edge_path', type=str, default='./init_edge_results')
parser.add_argument('--test_name', type=str, default='model')

# save settings
parser.add_argument('--save_path', type=str, default='./save_path/model/', help='the path to save model and logs')

# architecture settings
parser.add_argument('--backbone', type=str, default='efficientnet', choices=['resnet', 'res2net', 'mobilenet', 'efficientnet', 'ghostnet', 'pvt'])
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
parser.add_argument('--ratio', type=float, default=0.5)

# loss settings
# parser.add_argument('--mask_loss', type=str, default='f3', choices=['L2', 'hard', 'bi', 'bas', 'f3'])
# parser.add_argument('--edge_loss', type=str, default='f3', choices=['L2', 'hard', 'bi', 'bas', 'f3'])

opt = parser.parse_args()
