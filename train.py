import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from sklearn import decomposition
from sklearn import manifold
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import loompy
import random
import torch.utils.data as data
from tensorboardX import SummaryWriter
import math
from sklearn.cluster import KMeans

def sphere_data(data):
    print("sphere_data", data.shape)
    # mean = np.expand_dims(np.mean(data, axis=1),axis=1)
    # print("mean.shape", mean.shape)
    # std = np.expand_dims(np.std(data, axis=1),axis=1) + 1e-4
    # print("std.shape", std.shape)
    print("mean", np.mean(data))
    print("std", np.std(data))
    # return (data - np.mean(data)) / np.std(data)
    return (data - 0.08985414976598326) / 1.4195666640784363

def get_average_cells(cells, clusters, end_cluster, num_to_average, randomize_averaging=False):
    r = list(range(len(clusters)))

    # There seems to be some numerical error when we average using a different order.
    if randomize_averaging:
        random.shuffle(r)    
    next_cell_average = None
    summed_so_far = 0.0
    for i in r:
        if clusters[i] == end_cluster:
            if summed_so_far == 0:
                next_cell_average = cells[i]
            else:
                next_cell_average = next_cell_average + cells[i]
            # print(next_cell_average[0][10:50])

            summed_so_far += 1.0
            if summed_so_far == num_to_average:
                break
    # print(next_cell_average[0][10:50])
    # print("about to divide by ", summed_so_far)
    next_cell_average = next_cell_average / summed_so_far
    # print("averaged", averaged_so_far, "values")
    return next_cell_average

class LoomDataset(data.Dataset):
    def __init__(self, loom_file_path, total_clusters=6):
        self.total_clusters = total_clusters
        self.ds = loompy.connect(loom_file_path)
        self.spliced = self.ds.layer["spliced"][:, :].astype(np.dtype(float))
        self.unspliced = self.ds.layer["unspliced"][:, :].astype(np.dtype(float))
        self.ambig = self.ds.layer["ambiguous"][:, :].astype(np.dtype(float))
        self.spliced = np.transpose(self.spliced)
        self.unspliced = np.transpose(self.unspliced)
        self.ambig = np.transpose(self.ambig)
        self.cells = np.stack((self.spliced, self.unspliced, self.ambig))
        ca = dict(self.ds.col_attrs.items())
        self.clusters = ca["Clusters"][:]
        print(self.unspliced.shape)
        print(self.cells.shape)
        self.cells = np.transpose(self.cells, (1, 0, 2))
        self.cells = sphere_data(self.cells)
        print(self.cells.shape)

        # for i in range(100):
        #     print(self.spliced[i][i], self.unspliced[i][i], self.ambig[i][i])
        #     print(self.cells[i, i])
        print("len cells", len(self.cells))
        print("shape cells[0]", self.cells[0].shape)

    def get_num_genes(self):
        return len(self.cells[0][0])

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        current_cluster = self.clusters[index]
        current_cluster_average = get_average_cells(self.cells, self.clusters, current_cluster, random.randint(1, 10), randomize_averaging=True)
        next_cluster = current_cluster + 1
        if next_cluster > self.total_clusters:
            next_cluster = current_cluster
        next_cell_average = get_average_cells(self.cells, self.clusters, next_cluster, random.randint(1, 10), randomize_averaging=True)
        # rows, cols = np.nonzero(self.cells[index])
        # print(self.cells[index][rows][cols])
        return torch.FloatTensor(current_cluster_average), torch.FloatTensor(next_cell_average)

class SimpleLinearAutoEncoder(nn.Module):
    def __init__(self, num_genes=100, emb_dim=32):
        super(SimpleLinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 3, 100),
            nn.SELU(),
            nn.Conv1d(3, 1, 1),
            nn.SELU(),
            nn.Linear(num_genes-99, emb_dim),
            nn.SELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, num_genes-99),
            nn.SELU(),
            nn.ConvTranspose1d(1, 3, 1),
            nn.SELU(),
            nn.ConvTranspose1d(3, 3, 100),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SELUSimpleLinearAutoEncoder(nn.Module):
    def __init__(self):
        super(SELUSimpleLinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 8, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(8, 1, 101, padding=50),
            nn.SELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, 101, padding=50),
            nn.SELU(),
            nn.ConvTranspose1d(8, 3, 101, padding=50),
            nn.SELU(),
        )

    def forward(self, orig_x):
        x = self.encoder(orig_x)
        x = self.decoder(torch.cat((orig_x, x), dim=1))
        return x


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(16, 8, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(8, 8, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(8, 1, 101, padding=50),
            nn.SELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(4, 8, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(8, 8, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(8, 8, 101, padding=50),
            nn.SELU(),
            nn.Conv1d(8, 3, 101, padding=50),
            nn.SELU(),
        )

    def forward(self, orig_x):
        x = self.encoder(orig_x)
        x = self.decoder(torch.cat((orig_x, x), dim=1))
        return x
# torch.nn.Conv1d(1, 1, 15, 15)(torch.nn.Conv1d(1, 1, 10, 3)(torch.nn.Conv1d(3, 1, 1)


def display_heatmap(cells, clusters, decoder_output, start_cluster, end_cluster):
    fig=plt.figure(figsize=(3, 1))
    current_i = 0
    r = list(range(len(clusters)))
    random.shuffle(r)
    for i in r:
        if clusters[i] == start_cluster:
            cell_cpu = cells[i].cpu().detach().numpy()
            cell_cpu = np.repeat(cell_cpu, 1000, axis=0)
            print(cell_cpu.shape)
            ax = fig.add_subplot(3, 1, 1)
            ax.set_title(f'cluster {clusters[i]}')
            plt.imshow(cell_cpu, cmap='hot', interpolation='nearest', norm=matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0, clip=True))
            current_i = i
            break

    random.shuffle(r)    
    next_cell_average = get_average_cells(cells, clusters, end_cluster, 2.0, randomize_averaging=True)

    cell_cpu = next_cell_average.cpu().detach().numpy()
    cell_cpu = np.repeat(cell_cpu, 1000, axis=0)
    print(cell_cpu.shape)
    ax = fig.add_subplot(3, 1, 2)
    ax.set_title(f'average cluster {end_cluster}')
    plt.imshow(cell_cpu, cmap='hot', interpolation='nearest',  norm=matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0, clip=True))

    cell_cpu = decoder_output[current_i].cpu().detach().numpy()
    cell_cpu = np.repeat(cell_cpu, 1000, axis=0)
    ax = fig.add_subplot(3, 1, 3)
    ax.set_title(f'predicted')

    plt.imshow(cell_cpu, cmap='hot', interpolation='nearest',  norm=matplotlib.colors.Normalize(vmin=-2.0, vmax=2.0, clip=True))

def get_ground_truth_cluster_order(dataset):
    if "hgForebrainGlut" in dataset:
        return {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: None
        }
    elif "DentateGyrus" in dataset:
        # 6->7->10
        # 6->8->11->12
        # 6->9->3->5->4
        # 6->9->0->1
        # 6->9->0->2
        return {
            6: None,
            7: 10,
            10: None,
            8: 11,
            11: 12,
            12: None,
            9: None,
            3: 5,
            5: 4,
            4: None,
            0: None,
            1: None,
            2: None,
            13: None,
        }

def display_plot(args, model):
    ds = loompy.connect(args.dataset)
    spliced = ds.layer["spliced"][:, :]
    unspliced = ds.layer["unspliced"][:, :]
    ambig = ds.layer["ambiguous"][:, :]
    ca = dict(ds.col_attrs.items())
    spliced = np.transpose(spliced)
    unspliced = np.transpose(unspliced)
    ambig = np.transpose(ambig)
    spliced = spliced + unspliced + ambig
    print(spliced.shape)
    clusters = ca["Clusters"][:]
    print("clusters.shape", clusters.shape)
    print(ca.keys())
    colors = clusters
    set_for_dimen_reduction = spliced
    model = model.eval()
    quiver_ends = None

    if args.use_autoencoder:
        spliced = ds.layer["spliced"][:, :].astype(np.dtype(float))
        unspliced = ds.layer["unspliced"][:, :].astype(np.dtype(float))
        ambig = ds.layer["ambiguous"][:, :].astype(np.dtype(float))
        spliced = np.transpose(spliced)
        unspliced = np.transpose(unspliced)
        ambig = np.transpose(ambig)
        cells = np.stack((spliced, unspliced, ambig))
        cells = np.transpose(cells, (1, 0, 2))
        if args.model_type == "linear":
            cells = np.resize(cells, (cells.shape[0], cells.shape[1], 32738))


        cells = sphere_data(cells)
        cells = cells[:args.max_count]
        colors = clusters[:args.max_count]
        clusters = clusters[:args.max_count]
        cells = torch.FloatTensor(cells)
        cells = cells.cuda()

        with torch.no_grad():
            encoder_output = model.encoder(cells)
            if args.model_type == 'selu-linear' or args.model_type == 'cnn':
                decoder_output = model.decoder(torch.cat((cells, encoder_output), dim=1))
            else:
                decoder_output = model.decoder(encoder_output)
            encoder_output_next = model.encoder(decoder_output)
            encoder_output_numpy = encoder_output.squeeze(1).cpu().detach().numpy()
            set_for_dimen_reduction = encoder_output_numpy
            quiver_ends = encoder_output_next.squeeze(1).cpu().detach().numpy()
        if args.include_heatmap:
            display_heatmap(cells, clusters, decoder_output, 0, 1)
            display_heatmap(cells, clusters, decoder_output, 1, 2)
            display_heatmap(cells, clusters, decoder_output, 2, 3)
            display_heatmap(cells, clusters, decoder_output, 3, 4)
            display_heatmap(cells, clusters, decoder_output, 4, 5)
            display_heatmap(cells, clusters, decoder_output, 5, 6)

        plt.show()
    if args.type == "pca":
        pca = decomposition.PCA(n_components=args.dims)
        if args.use_autoencoder:
            total_points = np.concatenate((set_for_dimen_reduction, quiver_ends),axis=0)
            pca.fit(set_for_dimen_reduction)
            total_D = pca.transform(total_points)
            D = total_D[:len(set_for_dimen_reduction)]
            quiver_D = total_D[len(set_for_dimen_reduction):]
            arrows = quiver_D - D
        else:
            pca.fit(set_for_dimen_reduction)
            D = pca.transform(set_for_dimen_reduction)
    elif args.type == "svd":
        svd = decomposition.TruncatedSVD(n_components=args.dims)
        if args.use_autoencoder:
            total_points = np.concatenate((set_for_dimen_reduction, quiver_ends),axis=0)
            svd.fit(set_for_dimen_reduction)
            total_D = svd.transform(total_points)
            D = total_D[:len(set_for_dimen_reduction)]
            quiver_D = total_D[len(set_for_dimen_reduction):]
            arrows = quiver_D - D
        else:
            svd.fit(set_for_dimen_reduction)
            D = svd.transform(set_for_dimen_reduction)
    elif args.type == "ica":
        ica = decomposition.FastICA(n_components=args.dims)
        if args.use_autoencoder:
            total_points = np.concatenate((set_for_dimen_reduction, quiver_ends),axis=0)
            ica.fit(set_for_dimen_reduction)
            total_D = ica.transform(total_points)
            D = total_D[:len(set_for_dimen_reduction)]
            quiver_D = total_D[len(set_for_dimen_reduction):]
            arrows = quiver_D - D
        else:
            ica.fit(set_for_dimen_reduction)
            D = ica.transform(set_for_dimen_reduction)
    elif args.type == "tsne":
        tsne = manifold.TSNE(n_components=args.dims)
        if args.use_autoencoder:
            total_points = np.concatenate((set_for_dimen_reduction, quiver_ends),axis=0)

            total_D = tsne.fit_transform(total_points)
            D = total_D[:len(set_for_dimen_reduction)]
            quiver_D = total_D[len(set_for_dimen_reduction):]
            arrows = (quiver_D - D) / np.linalg.norm(arrows)
        else:
            D = tsne.fit_transform(set_for_dimen_reduction)

    if args.dims == 3:
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        c = np.arctan2(quiver_D[:, 0], quiver_D[:, 2])
        # Flatten and normalize
        c = (c.ravel() - c.min()) / c.ptp()
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.hsv(c)
        ax.quiver(D[:1000, 0], D[:1000, 1], D[:1000, 2], quiver_D[:1000, 0],  quiver_D[:1000, 1],  quiver_D[:1000, 2], colors=c[:1000], arrow_length_ratio=0.7, length=10, normalize=True) #cmap=plt.cm.nipy_spectral, edgecolor='k')
        ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=colors, cmap=plt.cm.nipy_spectral, edgecolor='k')
    else:
        fig, ax = plt.subplots()
        if args.arrows:
            ax.quiver(D[:, 0], D[:, 1], arrows[:, 0],  arrows[:, 1], headaxislength=7, headlength=11, headwidth=8, linewidths=0.25, width=0.00045, color=plt.cm.Dark2(colors), alpha=1)
            # ax.scatter(quiver_D[:, 0], quiver_D[:, 1], c=colors, cmap=plt.cm.Dark2, label=colors, edgecolor='k')
            # ax.scatter(D[:, 0], D[:, 1], c=colors, cmap=plt.cm.Dark2, label=colors, edgecolor='k')
        else:
            ax.scatter(D[:, 0], D[:, 1], c=colors, cmap=plt.cm.Dark2, label=colors, edgecolor='k')

        x = {}
        y = {}
        count = {}
        cluster_name = {}
        for i in range(len(colors)):
            cluster = colors[i]
            x[cluster] = 0.0
            y[cluster] = 0.0
            count[cluster] = 0.0
            if "ClusterName" in ca.keys():
                cluster_name[cluster] = ca["ClusterName"][i]

        for i in range(len(colors)):
            cluster = colors[i]
            x[cluster] += D[i, 0]
            y[cluster] += D[i, 1]
            count[cluster] += 1.0

        for c in x.keys():
            x[c] = x[c] / count[c]
            y[c] = y[c] / count[c]
            if c in cluster_name:
                plt.text(x[c], y[c], f"{c} {cluster_name[c]}", fontsize=13, bbox={"facecolor":"w", "alpha":0.6})
            else:
                plt.text(x[c], y[c], f"{c}", fontsize=13, bbox={"facecolor":"w", "alpha":0.6})
        gt = get_ground_truth_cluster_order(args.dataset)
        average_angle = 0.0
        count = 0.0
        for i in range(len(clusters)):
            gt_c = gt[clusters[i]]
            if gt_c is None:
                continue
            x_gt_c = x[gt_c]
            y_gt_c = y[gt_c]
            gt_arrow = np.array([x_gt_c - D[i, 0], y_gt_c - D[i, 1]])
            unit_vector_1 = gt_arrow / np.linalg.norm(gt_arrow)
            unit_vector_2 = arrows[i] / np.linalg.norm(arrows[i])
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)
            angle = angle * (180.0/math.pi)
            # print(i)
            # print(angle)
            average_angle += angle
            count += 1.0
        average_angle = average_angle / count
        print ("average angle: ", average_angle)
        kmeans = KMeans(n_clusters=len(x), random_state=0).fit(D)
        print(len(kmeans.labels_))
        # calculate cluster purity:
        assigned_cluster_counts = {}
        for i in range(len(clusters)):
            c = clusters[i]
            assigned_cluster = kmeans.labels_[i]
            if assigned_cluster not in assigned_cluster_counts:
                assigned_cluster_counts[assigned_cluster] = {}
            if c not in assigned_cluster_counts[assigned_cluster]:
                assigned_cluster_counts[assigned_cluster][c] = 1.0
            else:
                assigned_cluster_counts[assigned_cluster][c] += 1.0

        majority_cluster = {}
        for assigned_cluster in assigned_cluster_counts.keys():
            largest_count = 0.0
            for c, count in  assigned_cluster_counts[assigned_cluster].items():
                if count > largest_count:
                    majority_cluster[assigned_cluster] = c
                    largest_count = count
        print(majority_cluster)
        correct = 0.0
        for i in range(len(clusters)):
            if majority_cluster[kmeans.labels_[i]] == clusters[i]:
                correct += 1.0
        print("purity", correct / len(clusters))

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('--dims', type=int, default=3,
                        help="reduce dims to this amount")
    parser.add_argument('--count', type=int, default=1000,
                        help="amount of embeddings to plot")
    parser.add_argument('--type', type=str, default="pca",
                        help="pca or tsne or autoencoder")
    parser.add_argument('--use_autoencoder', action='store_true')
    parser.add_argument('--mode', type=str, default="train",
                        help="train or eval")
    parser.add_argument('--arrows', action='store_true')
    parser.add_argument('--dataset', type=str, default="./hgForebrainGlut.loom")
    parser.add_argument('--model_type', type=str, default='cnn')
    parser.add_argument('--include_heatmap', action='store_true')
    parser.add_argument('--max_count', type=int, default=1720)


    args = parser.parse_args()
    # ds = loompy.connect("/Users/kareem/Downloads/hgForebrainGlut.loom")
    dataset = LoomDataset(args.dataset)
    print("num_genes", dataset.get_num_genes())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=10)

    if args.model_type == 'cnn':
        model = CNNAutoEncoder()
    elif args.model_type == 'linear':
        model = SimpleLinearAutoEncoder(num_genes=32738, emb_dim=512)
    elif args.model_type == 'selu-linear':
        model = SELUSimpleLinearAutoEncoder()

    if torch.cuda.is_available():
        model = model.cuda()

    if args.checkpoint_path is not None:
        print("Resuming from checkpoint: %s" % args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint)
        model = model.cpu()
        save_path = args.checkpoint_path + ".cpu"
        torch.save(model.state_dict(), save_path)
        if torch.cuda.is_available():
           model = model.cuda()

    if args.mode == "eval":
        display_plot(args, model)
        sys.exit(0)

    tb_writer = SummaryWriter()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = nn.MSELoss()
    loss_steps = 10000
    checkpoint_steps = 10000
    pretrain_epochs = 8
    count = 0
    sum_loss = 0.0
    for epoch in range(0, 10000):
        for example in trainloader:
            cell, next_cell = example
            if torch.cuda.is_available():
                cell = cell.cuda()
                next_cell = next_cell.cuda()
            out = model(cell)
            l = loss(out, next_cell)
            sum_loss += l.item()
            if count % loss_steps == 0 and count != 0:
                print(sum_loss/loss_steps)
                tb_writer.add_scalars('loss',
                    {
                        "training": sum_loss/loss_steps,
                    },
                    count)
                sum_loss = 0.0
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            count += 1
            if count % checkpoint_steps == 0:
                save_path = os.path.join("./checkpoints/", 'chkpt_%d.pt' % count)
                torch.save(model.state_dict(), save_path)


