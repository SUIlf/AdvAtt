import torch
from copy import deepcopy
from util.utility import progress_bar, plot

from util.model_parser import parse_model


def run_epoch(model, dataloader, device, criterion, optimizer=None, is_training=False):
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if is_training:
            optimizer.zero_grad()
            
        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), f'Loss: {running_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
    
    return running_loss / len(dataloader), 100. * correct / total

def train_model(model, criterion, optimizer, trainloader, valloader, save_folder, epochs=10, device='cuda', logger=None):
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*10000)

    for epoch in range(epochs):
        print('\nEpoch: %d - LR: %.5f' % (epoch, scheduler.get_last_lr()[0]))
        
        if logger:
            logger.log(f'\nEpoch: {epoch + 1}/{epochs} - LR: {scheduler.get_last_lr()[0]:.5f}')
        
        train_loss, train_acc = run_epoch(model, trainloader, device, criterion, optimizer, is_training=True)
        if logger:
            logger.log(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}%')
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc = run_epoch(model, valloader, device, criterion)
        if logger:
            logger.log(f'Validation Loss: {val_loss:.3f}, Validation Acc: {val_acc:.3f}%')
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            if logger:
                logger.log('Saving improved model..')
            best_acc = val_acc
            print('Saving improved model..')
            best_model_wts = deepcopy(model.state_dict())
            torch.save({'model_state_dict': best_model_wts, 'acc': best_acc, 'epoch': epoch}, f'{save_folder}best_model.pth')
        
        scheduler.step()
        plot(history, epoch, save_folder)  # 更新训练和验证的历史记录图
    
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloader, device, criterion, logger=None):
    model.eval()
    test_loss, test_acc = run_epoch(model, dataloader, device, criterion, is_training=False)

    if logger:
        logger.log(f'Test Accuracy: {test_acc:.3f}%')
    print(f'Test Accuracy: {test_acc:.3f}%')


def model_load(dataset='cifar10', model_type='resnet18', model_path='./checkpoint', normalize=None):
    
    model = parse_model(dataset=dataset, model_type=model_type, normalize=normalize)
    save_folder = f'{model_path}/{dataset}/{model_type}/best_model.pth'
    # 加载保存的权重
    checkpoint = torch.load(save_folder)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model