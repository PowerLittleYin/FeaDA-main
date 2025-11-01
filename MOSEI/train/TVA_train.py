import torch
import datetime
import sys
import MOSEI.config as default_config
from MOSEI.data_loader import MOSEIDataloader
from tqdm import tqdm
import MOSEI.config as config

sys.path.append('./')
from MOSEI.utils import write_log, set_random_seed, write_config, Metrics
from MOSEI.models.model import TVA_fusion


def TVA_train_fusion(config, metrics, seed, train_data, valid_data):
    print('---------------TVA_EXP---------------')

    set_random_seed(seed)
    log_path = config.LOGPATH + "MOSEI_TVA_Test." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'
    update_epochs = config.MOSEI.downStream.update_epochs

    text_lr = config.MOSEI.downStream.TVAtrain.text_lr
    audio_lr = config.MOSEI.downStream.TVAtrain.audio_lr
    vision_lr = config.MOSEI.downStream.TVAtrain.vision_lr
    other_lr = config.MOSEI.downStream.TVAtrain.other_lr

    text_decay = config.MOSEI.downStream.TVAtrain.text_decay
    audio_decay = config.MOSEI.downStream.TVAtrain.audio_decay
    vision_decay = config.MOSEI.downStream.TVAtrain.vision_decay
    other_decay = config.MOSEI.downStream.TVAtrain.other_decay

    # 初始化模型
    model = TVA_fusion(config)

    # 设置多GPU训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(config.DEVICE)

    print(model)

    # 调用 load_froze 方法时需要访问原始模型
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_froze()
    else:
        model.load_froze()

    # 获取模型参数
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    text_params = list(model.proj_t.named_parameters()) + list(model.text_encoder.named_parameters())
    text_params = [p for _, p in text_params]
    vision_params = list(model.proj_v.named_parameters()) + list(model.vision_with_text.named_parameters())
    vision_params = [p for _, p in vision_params] + [model.promptv_m]
    # vision_params = [p for _, p in vision_params]
    audio_params = list(model.proj_a.named_parameters()) + list(model.audio_with_text.named_parameters())
    audio_params = [p for _, p in audio_params] + [model.prompta_m]
    # audio_params = [p for _, p in audio_params]
    model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n]

    optimizer_grouped_parameters = [
        {'params': text_params, 'weight_decay': text_decay, 'lr': text_lr},
        {'params': audio_params, 'weight_decay': audio_decay, 'lr': audio_lr},
        {'params': vision_params, 'weight_decay': vision_decay, 'lr': vision_lr},
        {'params': model_params_other, 'weight_decay': other_decay, 'lr': other_lr}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)

    loss, best_loss = 0, 1e8
    loss_a = loss_v = pred_loss = loss_nce = 0
    device = config.DEVICE
    total_epoch = config.MOSEI.downStream.TVAtrain.epoch
    mono_task_loss = pred_loss = sup_const_loss = 0
    best_epoch = 1

    for epoch in range(1, total_epoch + 1):
        model.train()
        left_epochs = update_epochs
        bar = tqdm(train_data, disable=False)
        for index, sample1 in enumerate(bar):
            for key in sample1:
                if isinstance(sample1[key], torch.Tensor):
                    sample1[key] = sample1[key].to(device)
            try:
                bar.set_description("Epoch:%d|loss:%s|pred_loss:%s|sup_const_loss:%s|loss_nce:%s" % (
                    epoch, loss.item(), pred_loss.item(), sup_const_loss.item(), loss_nce.item()
                )
                                    )
            except:
                bar.set_description(
                    "Epoch:%d|loss:%s|pred_loss:%s|sup_const_loss:%s|loss_nce:%s" % (
                        epoch, loss, pred_loss, sup_const_loss, loss_nce)
                )
            if left_epochs == update_epochs:
                optimizer.zero_grad()
            left_epochs -= 1
            idx = sample1['index']
            sample2 = train_data.dataset.sample(idx)
            for key in sample2:
                if isinstance(sample2[key], torch.Tensor):
                    sample2[key] = sample2[key].to(device)
            label = sample1['labels']['M'].clone().detach().to(device)

            pred, loss, loss_nce, pred_loss, sup_const_loss = model(sample1, sample2, mode='train')
            loss.backward()

            if not left_epochs:
                optimizer.step()
                left_epochs = update_epochs

        if not left_epochs:
            optimizer.step()

        print("EVAL valid")
        result, result_loss = eval(model, metrics, valid_data, device)
        print("result_loss is：" + str(result_loss))

        if result_loss < best_loss:
        # if epoch%3==0:
            best_loss = result_loss
            # 保存模型时需要保存的是原模型，而不是DataParallel包装后的模型
            if isinstance(model, torch.nn.DataParallel):
                model.module.save_model()
            else:
                model.save_model()


def TVA_test_fusion(config, metric, test_data):

    seed = config.seed
    log_path = config.LOGPATH + "MOSEI_TVA_Test." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'

    write_config(config, log_path)

    model = TVA_fusion(config=config)

    device = config.DEVICE
    model.to(device)


    model.load_model()



    result, loss = eval(model,metric, test_data, device)

    log = '\nTVA_Test\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2:%s\n\t' \
        'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
        '------------------------------------------' % (
        result['Has0_acc_2'], result['Has0_F1_score'], result['Non0_acc_2'], result['Non0_F1_score'],
        result['Mult_acc_5'], result['Mult_acc_7'], result['MAE'], result['Corr'], loss
    )
    print(log)
    write_log(log, log_path)
    return result


def eval(model, metrics, eval_data, device):
    with torch.no_grad():
        model.eval()
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            label = sample['labels']['M'].clone().detach().to(device).float()
            _pred, _all_loss, _loss_nce, _loss, _sup_const_loss = model(sample, None, mode="train")
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * 32
        pred = torch.cat(pred).to(torch.device('cpu')).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / len(eval_data)
        model.train()
    return eval_results, loss / len(eval_data)