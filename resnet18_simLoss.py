import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import numpy as np
import argparse
import csv
import sys

def generate_target_sim(sim, topk=5, gap=0, power=1):
    target_matrix = np.zeros((1000,1000))
    for i in range(1000):
        top_k = sim[i][np.argpartition(sim[i],-topk)[-topk:-topk+1]]
        target_matrix[i,i]=1
        for j in range(i):
            target_matrix[i,j] = pow(max(0,sim[i,j]-gap)/(1-gap), power) if sim[i,j]>=top_k and sim[i,j]>=gap else 0
            target_matrix[j,i] = target_matrix[i,j]
    return target_matrix

    
def train(train_loader, model, criterion, optimizer, num, accumulation_steps, use_gpu):
    losses = 0
    right_sum = 0
    jcn,wp = 0,0
    trained_num = 0

    # switch to train mode
    model.train()

    pbar = tqdm.tqdm(train_loader, leave=False, position=0)
    
    for i, (images, target) in enumerate(pbar):

        # move data to the same device as model
        
        if use_gpu:
            images = images.cuda(device=device_ids[args.device_id])
            target = target.cuda(device=device_ids[args.device_id])
        label = target_matrix[target]
        

        output= model(images)
        pred = torch.max(output, 1)[1]

        trained_num += target.shape[0]

        loss = criterion(output, label)
        if torch.isnan(loss):
            print('loss is nan')
            sys.exit()
        losses += loss.item()

        train_correct = (pred == target).sum()
        djcn,dwp = jcn_wp_sum(target, pred, jcn_matrix, wp_matrix)
        jcn += djcn
        wp += dwp
        right_sum += train_correct.item()
        pbar.set_postfix({"loss": '{:.5f}'.format(loss.item()/args.batch_size),"acc":'{:.5f}'.format(right_sum/trained_num)})
        loss.backward()
        if (i+1) % accumulation_steps == 0 or i+1 == num:
            optimizer.step()
            optimizer.zero_grad()       
    
    acc = right_sum/num
    
    return losses/num, acc, jcn/num, wp/num

def jcn_wp_sum(real, pred, jcn_matrix, wp_matrix):
    return torch.sum(jcn_matrix[real, pred]).item(),torch.sum(wp_matrix[real, pred]).item()

def validate(val_loader, model, criterion, num, use_gpu):
    losses=0
    right_sum = 0
    jcn,wp = 0,0
    
    model.eval()
    with torch.no_grad():    
        for i, (images, target) in enumerate(tqdm.tqdm(val_loader, leave=False, position=0)):
            
            if use_gpu:
                images = images.cuda(device=device_ids[args.device_id])
                target = target.cuda(device=device_ids[args.device_id])
            label = target_matrix[target]
            
            output= model(images)
            pred = torch.max(output, 1)[1]
            train_correct = (pred == target).sum()
            djcn,dwp = jcn_wp_sum(target, pred, jcn_matrix, wp_matrix)
            jcn += djcn
            wp += dwp
            right_sum += train_correct.item()
            losses += criterion(output, label).item()
                        
    acc = right_sum/num
  
    return losses/num, acc, jcn/num, wp/num

def load_data(batch_size):

    transforms_train = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    imagenet_data_train = torchvision.datasets.ImageFolder(root='../../../../local/scratch/datasets/ImageNet/ILSVRC2012/train', transform = transforms_train)
    train_data_loader = torch.utils.data.DataLoader(imagenet_data_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    transforms_validation = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    imagenet_data_validation = torchvision.datasets.ImageFolder(root='../../../../local/scratch/datasets/ImageNet/ILSVRC2012/val', transform = transforms_validation)
    val_data_loader = torch.utils.data.DataLoader(imagenet_data_validation, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return imagenet_data_train,train_data_loader,imagenet_data_validation,val_data_loader

class EarlyStopping():
    def __init__(self, tolerence = 10, type='loss'):
        if type not in ['loss', 'acc']:
            assert("The type should be 'loss' or 'acc'")
        self.type = type
        self.best_score = np.Inf if type == 'loss' else 0
        self.tolerence = tolerence
        self.count_tolerence = tolerence
        self.best_model = None
    
    def check(self, value, model):
        if (value <= self.best_score and self.type == 'loss') or (value >= self.best_score and self.type == 'acc'):
            self.count_tolerence = self.tolerence
            self.best_score = value
            self.best_model = model
        else:
            self.count_tolerence -= 1
            if self.count_tolerence < 0:
                print('Early Stopping!')
                return True, self.best_model
        return False, None
    
class customLoss(torch.nn.Module):

    def __init__(self, lamda=1):
        super(customLoss, self).__init__()
        self.lamda = lamda
        
    def forward(self, output, label):
        probs = torch.nn.functional.softmax(output,dim=1)
        weighted_probs = torch.multiply(probs, label)
        loss = -torch.sum(torch.log(torch.sum(weighted_probs,dim=1)))
        return loss
        #return CE



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Training parameters')
    parser.add_argument('--batch_size', default = 128, type=int, help ='Batch size')
    parser.add_argument('--accumulation_steps', default =1, type=int, help = 'Accumulation step')
    parser.add_argument('--lr', default = 1e-3, type=float, help='Learning rate')
    parser.add_argument('--momentum', default = 0.9, type=float, help='Momnetum of SGD')
    parser.add_argument('--use_gpu', default = True, type=bool, help='Whether to use gpu to train')
    parser.add_argument('--device_id', default = 0, type=int, help ='GPU ID')
    parser.add_argument('--start_epoch', default = 0, type=int, help='Number of start training epoch, from 1')
    parser.add_argument('--end_epochs', default = 100, type=int, help='Number of training epoch')
    parser.add_argument('--parallel',default = False, type=bool, help='Whether use parallel to train')
    parser.add_argument('--name',default = 'resnet18_simLoss_09_3', type=str, help='Name of the saved file')
    args = parser.parse_args()

    jcn_matrix = np.load('data/jcn_similarity_matrix.npy')
    wp_matrix = np.load('data/wu_palmer_similarity_matrix.npy')
    target_matrix = torch.from_numpy(generate_target_sim(wp_matrix, topk=5, gap=0.9, power = 3))
    #target_matrix = torch.eye(1000)
    jcn_matrix = torch.from_numpy(jcn_matrix)
    wp_matrix = torch.from_numpy(wp_matrix)


    imagenet_data_train,train_data_loader,imagenet_data_validation,val_data_loader = load_data(args.batch_size)

    device_ids=[0,1,2,3,4,5,6,7]

    resnet18 = torchvision.models.resnet18(weights = None)
    model = resnet18
    criterion = customLoss()
    if args.use_gpu:
        if args.parallel:
            model = torch.nn.DataParallel(model).cuda(device=device_ids[args.device_id])
        else:
            model.cuda(device=device_ids[args.device_id])
        criterion = criterion.cuda(device=device_ids[args.device_id])
        jcn_matrix = jcn_matrix.cuda(device=device_ids[args.device_id])
        wp_matrix = wp_matrix.cuda(device=device_ids[args.device_id])
        target_matrix = target_matrix.cuda(device=device_ids[args.device_id])
    else:
        model = model.cpu()
        criterion = criterion.cpu()

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr , momentum = args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    earlystopping = EarlyStopping(type = 'acc')

    if os.path.isfile('log/log_'+args.name+'.csv'):
        if input("Overwriting '"+'log/log_'+args.name+'.csv'+"', type 'y' to continue...") != 'y':
            sys.exit()
    with open('log/log_'+args.name+'.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch','Train_loss','Train_acc','Train_jcn','Train_wp',"Val_loss",'Val_acc','Val_jcn','Val_wp'])
    for epoch in range(args.start_epoch, args.end_epochs):
        train_loss, train_acc, train_jcn_avg, train_wp_avg = train(train_data_loader, model, criterion, optimizer, len(imagenet_data_train), args.accumulation_steps, args.use_gpu)
        val_loss, val_acc, val_jcn_avg, val_wp_avg = validate(val_data_loader, model, criterion, len(imagenet_data_validation), args.use_gpu)
        scheduler.step(val_loss)

        with open('log/log_'+args.name+'.csv', 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch+1,train_loss,train_acc,train_jcn_avg,train_wp_avg,val_loss,val_acc,val_jcn_avg,val_wp_avg])
        info_str = 'Epoch '+str(epoch+1)+\
            ': train_loss: '+str(train_loss)+' train_accuarcy: '+str(train_acc)+' train_jcn_avg: '+str(train_jcn_avg)+\
            ' validation_loss: '+str(val_loss)+' validation_accuarcy: '+str(val_acc)+' validation_jcn_avg: '+str(val_jcn_avg)
        if (epoch+1)%5==0:
            print(info_str)
        
        if (epoch+1)%20==0:
            checkpoint = {
			    'model': model.state_dict(),
			    'optimizer': optimizer.state_dict(),
			    'epoch': epoch,
                'accuarcy':val_acc
		    }
            torch.save(checkpoint, 'checkpoint/%s_%s.pth' % (args.name, str(epoch+1)))

        stop, best_model = earlystopping.check(val_wp_avg, model)
        if stop:
            print(info_str)
            checkpoint = {
			    'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
		    }
            torch.save(checkpoint, 'checkpoint/%s_best_model.pth' % (args.name))
            break

    print('Train finished')

    
