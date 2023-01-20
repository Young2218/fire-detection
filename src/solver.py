import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Solver:
    def __init__(self, model, max_epoch, early_stop, 
                 train_loader, val_loader, 
                 save_path, log_save_path,
                 optimizer,criterion) -> None:
        self.model = model
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.save_path = save_path
        
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_save_path)
    
        
    def train(self):
        self.model.to(self.device)
        min_val_loss = 500000
        check_early_stop = 0
        
        for epoch in range(1, self.max_epoch):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()
            
            self.writer.add_scalar("LOSS/train_loss", train_loss, epoch)
            self.writer.add_scalar("LOSS/val_loss", val_loss, epoch)
            # self.writer.add_scalar("f1_socre", val_f1, epoch)
            self.writer.flush()
            
            print(f'EPOCH[{epoch}] TRAIN_LOSS[{train_loss}] VAL_LOSS[{val_loss}]')
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                check_early_stop = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                check_early_stop += 1
                if check_early_stop > self.early_stop:
                    print("EARLY STOP")
                    self.writer.close()
                    break
        
        print(f"End Train in {epoch} epochs, Min Loss[{min_val_loss}]")
        return str(min_val_loss)
        
    def train_one_epoch(self) -> float:
        self.model.train()        
        loss_list = []
        for images, labels in tqdm(self.train_loader):
            # get the inputs
            images = images.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = self.model(images)
            # compute the loss based on model output and real labels
            loss = self.criterion(outputs, labels)
            loss_list.append(loss)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            self.optimizer.step()
        
        return sum(loss_list)/len(loss_list)

    def validate_one_epoch(self) -> float:
        self.model.eval()        
        loss_list = []
        f1_list = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                # get the inputs
                images = images.to(self.device)
                labels = labels.to(self.device)

                # predict classes using images from the training set
                outputs = self.model(images)
                # compute the loss based on model output and real labels
                loss = self.criterion(outputs, labels)
                loss_list.append(loss)
               
                # # for calculate f1 socre
                # f1 = self.f1_calculater(outputs, labels)
                # f1_list.append(f1)
                
        # f1_score = sum(f1_list)/len(f1_score)
        val_loss = sum(loss_list)/len(loss_list)
        return val_loss

    def inference(self):
        pass
    
    def load_model(self):
        pass