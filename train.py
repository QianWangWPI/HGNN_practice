# train.py
import argparse
import torch
import torch.nn.functional as F
from data_loader import load_dataset, build_han_data
from model_hgnn import GraphSAGE, HAN
from utils import plot_multiclass_roc
from sklearn.metrics import f1_score
def arg_parse():
    parser = argparse.ArgumentParser(description='Train HGNN on DBLP dataset')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--dataset', type=str, default='IMDB', help='DBLP, ACM, IMDB')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='training ratio')
    parser.add_argument('--model', type=str, default='HAN', help='GraphSAGE, HAN')

    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden layer dimensions')
    parser.add_argument('--out_channels', type=int, default=3, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--log_every', type=int, default=50, help='Log results every n epochs')
    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data,metadata,y,train_idx,val_idx,test_idx,target_ntype = load_dataset(args)
    args.target_ntype = target_ntype
    y = y.to(device)
    # train_idx = data['author'].train_mask.to(device)
    # val_idx = data['author'].val_mask.to(device)
    # test_idx = data['author'].test_mask.to(device)
    data = data.to(device)
    # Initialize model and optimizer
    if args.model == 'GraphSAGE':
        model = GraphSAGE(metadata, args).to(device)
    elif args.model == 'HAN':
        features_list, adj, author_y, train_mask, val_mask, test_mask = build_han_data(args)
        inputs_list = [x.to(device) for x in features_list]
        biases_list = [adj.to(device) for adj in adj]
        print(len(features_list[0][0]))
        model = HAN(num_metapaths=len(biases_list), in_dim=len(features_list[0][0]), hidden_dim=args.hidden_channels, out_dim=args.out_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        if args.model == 'GraphSAGE':
            out = model(data.x_dict, data.edge_index_dict)
        elif args.model == 'HAN':
            out, att = model(inputs_list, biases_list)
        # print(out.shape)
        # print(y)

        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            if args.model == 'GraphSAGE':
                out = model(data.x_dict, data.edge_index_dict)
            elif args.model == 'HAN':
                out, att = model(inputs_list, biases_list)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_idx] == y[val_idx]).sum().item() / val_idx.sum().item()
            test_acc = (pred[test_idx] == y[test_idx]).sum().item() / test_idx.sum().item()


        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            # AUC plot
            # plot_multiclass_roc(y[test_idx], F.softmax(out[test_idx], dim=1), num_classes=4)
            pred = out.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
            macro_f1 = f1_score(true[test_idx.cpu().numpy()], pred[test_idx.cpu().numpy()], average='macro')
            print(macro_f1)

if __name__ == '__main__':
    args = arg_parse()
    train(args)
