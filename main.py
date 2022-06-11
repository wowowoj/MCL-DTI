import os
import shutil
import torch
import argparse
from datetime import datetime
from model import MCLDTI
from Train import Train_model, Tester
from utils import data_loader, get_img_path

from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc, average_precision_score


def model_run(args):
    # time
    global m_path
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    run_time = datetime.now().strftime(ISOTIMEFORMAT)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"数据集：{args.dataset_name} 运行时间: {run_time}")

    resul_name = "result/" + args.dataset_name
    if not os.path.exists(resul_name):
        os.makedirs(resul_name)

    result_file = resul_name + "/" + run_time + \
                  " learnable " + \
                  " Tesing...  " + \
                  " lr_decay: " + str(args.lr_decay) + \
                  " Conv1d: 3L " + \
                  " Conv2d: 2L " + \
                  " Batchsize: " + str(args.batch_size) + \
                  " K: " + str(args.k) + \
                  " encoder_depth: " + str(args.depth_e1) + ", " + str(args.depth_e2) + \
                  " decoder_depth:  " + str(args.depth_decoder) + " .txt"

    # ********************************* Train_dataset *********************************
    train_img_path = "data/" + args.dataset_name + "/train/" + "Img_" + str(args.img_size) + "_" + str(
        args.img_size) + "/img_inf_data"
    train_image = get_img_path(train_img_path)
    train_smile_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_smiles_" + str(
        args.k) + "_gram.npy"
    train_pro_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_proteins_" + str(
        args.k) + "_gram.npy"
    train_inter_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_interactions.npy"
    train_dataset, train_loader = data_loader(batch_size=args.batch_size,
                                              imgs=train_image,
                                              smile_name=train_smile_name,
                                              pro_name=train_pro_name,
                                              inter_name=train_inter_name)

    # ********************************* Val_dataset *********************************
    val_img_path = "data/" + args.dataset_name + "/val/" + "Img_" + str(args.img_size) + "_" + str(
        args.img_size) + "/img_inf_data"
    val_image = get_img_path(val_img_path)

    val_smile_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_smiles_" + str(
        args.k) + "_gram.npy"
    val_pro_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_proteins_" + str(
        args.k) + "_gram.npy"
    val_inter_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_interactions.npy"
    val_dataset, val_loader = data_loader(batch_size=args.batch_size,
                                          imgs=val_image,
                                          smile_name=val_smile_name,
                                          pro_name=val_pro_name,
                                          inter_name=val_inter_name)

    # ********************************* Test_dataset *********************************
    test_img_path = "data/" + args.dataset_name + "/test/" + "Img_" + str(args.img_size) + "_" + str(
        args.img_size) + "/img_inf_data"
    test_image = get_img_path(test_img_path)

    test_smile_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_smiles_" + str(
        args.k) + "_gram.npy"
    test_pro_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_proteins_" + str(
        args.k) + "_gram.npy"
    test_inter_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_interactions.npy"
    test_dataset, test_loader = data_loader(batch_size=args.batch_size,
                                            imgs=test_image,
                                            smile_name=test_smile_name,
                                            pro_name=test_pro_name,
                                            inter_name=test_inter_name)

    # ********************************* create model *********************************
    protein_len=10
    if (args.dataset_name == "Human"):
        protein_len = 23
    elif (args.dataset_name == "Davis"):
        protein_len = 22
    elif (args.dataset_name == "Celegans"):
        protein_len = 21
    elif (args.dataset_name == "BindingDB"):
        protein_len = 41

    torch.manual_seed(2)
    model = MCLDTI(depth_e1=args.depth_e1,
                   depth_e2=args.depth_e2,
                   depth_decoder=args.depth_decoder,
                   embed_dim=args.embed_dim,
                   protein_dim=args.protein_dim,
                   drop_ratio=args.drop_ratio,
                   backbone=args.backbone,
                   protein_len=protein_len
                   ).to(device)

    # ********************************* training *********************************
    lr, lr_decay, weight_decay = map(float, [1e-3, args.lr_decay, 1e-8])

    trainer = Train_model(model, lr, weight_decay)

    print("开始训练....")
    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        total_loss = []

        for i, data_train in enumerate(train_loader):
            if data_train[0].shape[0] <= 1:
                break

            loss_train = trainer.train(data_train)
            total_loss.append(loss_train)
            if (i + 1) % 50 == 0:
                print(
                    f"Training "
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Batch  {i}/{len(train_loader)}] "
                    f"[batch_size {data_train[0].shape[0]}] "
                    f"[loss_train : {loss_train}]")

        # model save
        model_path = "data/" + args.dataset_name + "/output/model/"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_file = model_path + run_time + "_" + str(epoch) + ".model"
        torch.save(model, model_file)
        print("Epoch: ", epoch, "     Train avg_loss: ", sum(total_loss) / len(train_loader))

        # ********************************* Val , Test *********************************
        with torch.no_grad():
            val(args, model_file, val_loader)
            aucs = test(args, model_file, test_dataset, test_loader, result_file)


def val(args, file_model, val_loader):
    torch.manual_seed(2)
    model = torch.load(file_model)
    valer = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    for i, data_list in enumerate(val_loader):
        loss, correct_labels, predicted_labels, predicted_scores, _, _ = valer.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)

    loss_val = sum(Loss) / len(val_loader)
    AUC_val = roc_auc_score(y_label, y_score)
    fpr, tpr, thresholds = roc_curve(y_label, y_score)

    AUPRC = average_precision_score(y_label, y_score)

    precision_val = precision_score(y_label, y_pred)
    recall_val = recall_score(y_label, y_pred)
    f1_score = (2 * precision_val * recall_val) / (recall_val + precision_val + 0.0001)
    print(
        "Valing   batch_size %d [loss : %.3f] [AUC : %.3f] [AUPRC : %.3f] [precision : %.3f] [recall : %.3f] [F1 : %.3f] "
        % (args.batch_size, loss_val, AUC_val, AUPRC, precision_val, recall_val, f1_score)
    )


def test(args, file_model, test_dataset, test_loader, file_AUCs_test):
    torch.manual_seed(2)
    model = torch.load(file_model)
    tester = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    rate1s = []
    rate2s = []
    for i, data_list in enumerate(test_loader):
        loss, correct_labels, predicted_labels, predicted_scores, rate1, rate2 = tester.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)
        for r1 in rate1:
            rate1s.append(r1)
        for r2 in rate2:
            rate2s.append(r2)

    loss_test = sum(Loss) / len(test_loader)
    AUC_test = roc_auc_score(y_label, y_score)
    fpr, tpr, thresholds = roc_curve(y_label, y_score)

    AUPRC = average_precision_score(y_label, y_score)

    precision_test = precision_score(y_label, y_pred)
    recall_test = recall_score(y_label, y_pred)
    f1_score = (2 * precision_test * recall_test) / (recall_test + precision_test + 0.0001)

    rr1 = sum(rate1s) / len(rate1s)
    rr2 = sum(rate2s) / len(rate2s)
    print(
        "Testing  batch_size %d [loss : %.3f] [AUC : %.3f] [AUPRC : %.3f] [precision : %.3f] [recall : %.3f] [F1 : %.3f] [rate1: %.5f] [rate2: %.5f]"
        % (args.batch_size, loss_test, AUC_test, AUPRC, precision_test, recall_test, f1_score, rr1, rr2)
    )
    print()

    AUCs = [len(test_dataset),
            len(test_loader),
            format(loss_test, '.3f'),
            format(AUC_test, '.3f'),
            format(precision_test, '.3f'),
            format(recall_test, '.3f'),
            format(f1_score, ".3f"),
            format(AUPRC, '.3f'),
            format(rr1, '.5f'),
            format(rr2, ".5f")]
    tester.save_AUCs(AUCs, file_AUCs_test)
    return AUCs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="Davis")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="CNN")
    # dim
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--img_dim', type=int, default=2)
    parser.add_argument('--fingure_dim', type=int, default=64)
    parser.add_argument('--smile_dim', type=int, default=64)
    parser.add_argument('--protein_dim', type=int, default=256)

    # depth
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--depth_e1', type=int, default=1)
    parser.add_argument('--depth_e2', type=int, default=1)
    parser.add_argument('--depth_decoder', type=int, default=1)

    parser.add_argument('--lr_decay', type=float, default=0.85)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--device", default='cuda:0')
    opt = parser.parse_args()
    model_run(opt)
