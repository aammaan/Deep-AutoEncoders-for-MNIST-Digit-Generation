import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import matplotlib.image as mpimg
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pickle
from EncDec import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Shorten_Data():
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    def calculate_intensity_difference(self,image1, image2):
        """Calculates the average intensity difference between two grayscale images."""
        image1 = mpimg.imread(os.path.join(self.path,image1))
        image2 = mpimg.imread(os.path.join(self.path,image2))
        if len(image1.shape)==3:
            image1 = self.rgb2gray(image1)
        if len(image2.shape)==3:
            image2 = self.rgb2gray(image2)
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions")
        # Calculate absolute difference between pixel intensities
        difference = np.abs(image1 - image2)
        # Average intensity difference across all pixels
        return np.mean(difference)

    def find_top_pairs(self,intensity_differences, num_images ,num_pairs):
        """Finds the top 'num_pairs' image pairs with the maximum intensity difference."""
        # Flatten the matrix for easier sorting (optional)
        flattened_differences = intensity_differences.flatten()
        # Get indices of the highest intensity differences (excluding diagonal)
        sorted_indices = np.argsort(flattened_differences)[::-1][1:num_pairs+1]  # Avoid self-comparison
        # Convert indices back to image pair format ((image1_index, image2_index), ...)
        top_pairs = [(int(idx / num_images), idx % num_images) for idx in sorted_indices]
        return top_pairs

    def short(self,label_dict,number_of_pairs,path):
        super(Shorten_Data,self).__init__()
        # label_dict is input dictionary in which key is label and value contain the list of image names of that label
        new_label_dict = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'10':[]}
        self.path = path
        for i in tqdm(label_dict):
            num_images = len(label_dict[i])
            # print(num_images)
            # input("wait")
            intensity_differences = np.zeros((num_images, num_images), dtype=np.float32)
            for j in tqdm(range(num_images)):
                for k in range(j,num_images):
                    difference = self.calculate_intensity_difference(label_dict[i][j],label_dict[i][k])
                    intensity_differences[j,k] = difference
            top_pairs = self.find_top_pairs(intensity_differences,num_images, num_pairs=number_of_pairs)      
            for pair in top_pairs:
                new_label_dict[i].append(label_dict[i][pair[0]])
                new_label_dict[i].append(label_dict[i][pair[1]])
            new_label_dict[i] = list(np.unique(np.array(new_label_dict[i])))
        return new_label_dict

class AlteredMNIST(Dataset):
    """
    dataset description:
    
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    Write code to load Dataset
    """
    def __init__(self, aug_data_path = "Data/aug/"):
        self.data_path_aug = aug_data_path
        # self.data_path_clean = "../Data/clean/"
        # self.label_dict = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],}
        # for file_name in os.listdir(self.data_path_aug):
        #     _ , l = file_name.rsplit("_",1)
        #     label,_ = l.split('.')
        #     self.label_dict[label].append(file_name)

        self.train_data = [file_name for file_name in os.listdir(self.data_path_aug)]

        # self.train_data = self.load_dictionary('min_aug.unknown')
        # self.train_keys = list(self.train_data.keys())
        # random.shuffle(self.train_keys)
        # print('file_read_done')


        # self.shorten_data_object = Shorten_Data()
        # self.shorten_data = self.shorten_data_object.short(self.label_dict,1100,self.data_path_aug)
        # file = open('aug_data_dictionary_short', 'wb') 
        # pickle.dump(self.shorten_data, file) 
        # file.close() 
        # self.train_data = []
        # for i in self.shorten_data.keys():
        #     self.train_data = self.train_data + self.shorten_data[i]

    def load_dictionary(self,path):
        data={}
        with open(path, 'r') as file:
            # Read each line of the file
            for line in file:
                # Split the line into key and value parts
                key, value = line.strip().split(": ")
                # Convert the string representation of the list into an actual list
                value_list = eval(value)
                # Assign the key and value to the dictionary
                data[key] = value_list
            return data

    def __len__(self):
        # return len(self.train_data.keys())
        return len(self.train_data)
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    def __getitem__(self,index):
        # key = self.train_keys[index]
        _,l = self.train_data[index].rsplit("_",1)
        label,_ = l.split('.')
        # temp = mpimg.imread(os.path.join(self.data_path_aug,key))

        temp = mpimg.imread(os.path.join(self.data_path_aug,self.train_data[index]))
        if len(temp.shape)==3:
            temp = self.rgb2gray(temp)
        temp = torch.Tensor(temp)
        # temp = (temp/255)
        temp = temp.reshape(1,28,28)
        return {'item' : temp, 'label' : label}
        # return {'item' : temp, 'label' : self.train_data[key]}

class Encoder_Residual_Block(nn.Module):
    def __init__(self,in_channels , out_channels , mid_channels, kernel_size = 3, dilation = 3, padding = 0):
        super(Encoder_Residual_Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = mid_channels,
                               kernel_size = kernel_size,
                               dilation = dilation,
                               padding  = padding)
        self.conv2 = nn.Conv2d(in_channels = mid_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size,
                               dilation = dilation,
                               padding  = padding)
        self.batchnorm1 = nn.BatchNorm2d(num_features = mid_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(
                    self.batchnorm2(
                        self.conv2(
                            self.relu(
                                self.batchnorm1(
                                    self.conv1(x)
                                )
                            )
                        )
                    )
                )

class Encoder_Skip_Block(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1, stride = 2, padding = 2):
        super(Encoder_Skip_Block,self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding)
        self.batchnorm = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(
                    self.batchnorm(
                        self.conv(x)
                    )
                )

class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self):
        super(Encoder,self).__init__()
        self.block1_1 = Encoder_Residual_Block(in_channels=1,
                                               out_channels=32,
                                               mid_channels=16,
                                               kernel_size = 3,
                                               dilation = 3)
        self.skip1_1 = Encoder_Skip_Block(in_channels=1,
                                          out_channels=32)
        self.block2 = Encoder_Residual_Block(in_channels=32,
                                             out_channels=64,
                                             mid_channels=48,
                                             dilation = 1,
                                             padding = 1)
        self.skip2 = Encoder_Skip_Block(in_channels=32,
                                        out_channels=64,
                                        stride = 1,
                                        padding = 0)
        self.block3 = Encoder_Residual_Block(in_channels=64,
                                             out_channels=128,
                                             mid_channels=96,
                                             dilation = 2)
        self.skip3 = Encoder_Skip_Block(in_channels=64,
                                        out_channels=128,
                                        padding = 0)
        self.block4 = Encoder_Residual_Block(in_channels=128,
                                             out_channels=256,
                                             mid_channels=192,
                                             dilation = 1)
        self.skip4 = Encoder_Skip_Block(in_channels=128,
                                        out_channels=256,
                                        kernel_size=1,
                                        stride=2,
                                        padding=0)
        
        self.mean_vae = torch.nn.Linear(4096,2)
        self.log_var = torch.nn.Linear(4096,2)
        
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    
    
    def forward(self,x):
        # if(x.shape[1]==1):
        #     out = self.block1_1(x)
        #     skip_out = self.skip1_1(x)
        #     out = out+skip_out
        # else:
        #     out = self.block1_3(x)
        #     skip_out = self.skip1_3(x)
        #     out = out+skip_out
        out = self.block1_1(x)
        skip_out = self.skip1_1(x)
        out = out+skip_out

        out_ = self.block2(out)
        skip_out_ = self.skip2(out)
        out = out_+skip_out_

        out_ = self.block3(out)
        skip_out_ = self.skip3(out)
        out = out_+skip_out_

        out_ = self.block4(out)
        skip_out_ = self.skip4(out)
        out = out_+skip_out_

        vae_out = out.view(out.size(0),-1)
        mean = self.mean_vae(vae_out)
        log_var = self.log_var(vae_out)
        encoded_vae = self.reparameterize(mean,log_var)

        return out,mean, log_var, encoded_vae

class Decoder_Residual_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size,
                 dilation,
                 padding,
                 stride):
        super(Decoder_Residual_Block,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=mid_channels,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        stride=stride,
                                        padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(num_features=mid_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(in_channels=mid_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        stride=stride,
                                        padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self,x):
        return self.relu(
                    self.batchnorm2(
                        self.conv2(
                            self.relu(
                                self.batchnorm1(
                                    self.conv1(x)
                                )
                            )
                        )
                    )
                )

class Decoder_Skip_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 padding,
                 stride,
                 output_padding=0):
        super(Decoder_Skip_Block,self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       dilation=dilation,
                                       padding=padding,
                                       stride=stride,
                                       output_padding=output_padding)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(
                    self.batchnorm(
                        self.conv(x)
                    )
                )

class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self, in_channels = 256, out_channels = 1):
        super(Decoder,self).__init__()
        self.block1 = Decoder_Residual_Block(in_channels=256,
                                             out_channels=128,
                                             mid_channels=192,
                                             kernel_size=3,
                                             dilation=1,
                                             padding=0,
                                             stride=1)
        self.skip_block1 = Decoder_Skip_Block(in_channels=256,
                                              out_channels=128,
                                              kernel_size=1,
                                              dilation=1,
                                              padding=0,
                                              stride=2,
                                              output_padding=1)
        self.block2 = Decoder_Residual_Block(in_channels=128,
                                             out_channels=64,
                                             mid_channels=96,
                                             kernel_size=3,
                                             dilation=2,
                                             padding=0,
                                             stride=1)
        self.skip_block2 = Decoder_Skip_Block(in_channels=128,
                                              out_channels=64,
                                              kernel_size=1,
                                              dilation=1,
                                              stride=2,
                                              padding=0,
                                              output_padding=1)
        self.block3 = Decoder_Residual_Block(in_channels=64,
                                             out_channels=32,
                                             mid_channels=48,
                                             kernel_size=3,
                                             dilation=1,
                                             padding=1,
                                             stride=1)
        self.skip_block3 = Decoder_Skip_Block(in_channels=64,
                                              out_channels=32,
                                              kernel_size=1,
                                              dilation=1,
                                              padding=0,
                                              stride=1)
        self.block4 = Decoder_Residual_Block(in_channels=32,
                                             out_channels=1,
                                             mid_channels=16,
                                             kernel_size=3,
                                             dilation=3,
                                             padding=0,
                                             stride=1)
        self.skip_block4 = Decoder_Skip_Block(in_channels=32,
                                              out_channels=1,
                                              kernel_size=1,
                                              dilation=1,
                                              padding=2,
                                              stride=2,
                                              output_padding=1)
        self.linear = nn.Linear(2,4096)
        # self.sigmoid =nn.Sigmoid()
    
    def forward(self,x):
        if(len(x.shape)==2):
            x = self.linear(x)
            x = x.view(-1,256,4,4)

        out = self.block1(x)
        skip_out = self.skip_block1(x)
        out = out+skip_out

        out_ = self.block2(out)
        skip_out_ = self.skip_block2(out)
        out = out_+skip_out_

        out_ = self.block3(out)
        skip_out_ = self.skip_block3(out)
        out = out_+skip_out_

        out_ = self.block4(out)
        skip_out_ = self.skip_block4(out)
        out = out_+skip_out_

        # if(len(x.shape)==2):
        #     return self.sigmoid(out)

        return out
        
class AELossFn(nn.Module):
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self,clean_data_path = "Data/clean/"):
        super(AELossFn,self).__init__()
        self.mse_loss = nn.MSELoss()
        self.data_path_clean = clean_data_path
        # file_open = open('EncDec/clean_data_dictionary_short','rb')
        # self.label_dict = pickle.load(file_open)
        self.label_dict_all = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],}
        for file_name in os.listdir(self.data_path_clean):
            _ , l = file_name.rsplit("_",1)
            label_index,_ = l.split('.')
            self.label_dict_all[label_index].append(file_name)
        self.label_dict = self.label_dict_all
        # self.label_dict_object = Shorten_Data()
        # self.label_dict = self.label_dict_object.short(self.label_dict_all,50,self.data_path_clean)
        # file = open('clean_data_dictionary_short', 'wb') 
        # pickle.dump(self.label_dict, file) 
        # file.close()
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
        
    def forward(self,x_, label_):
        total_loss = 0.0
        avg_loss_image = 0.0
        total_ssim = 0.0
        avg_ssim_image = 0.0
        for x,label in zip(x_,label_):
            # label_image_paths = label
            label_image_paths = self.label_dict[label]
            avg_loss_image = 0.0
            avg_ssim_image = 0.0
            for i,path in enumerate(label_image_paths):
                if(i==50):
                    break
                path = os.path.join(self.data_path_clean,path)
                temp = mpimg.imread(path)
                if len(temp.shape)==3:
                    temp = self.rgb2gray(temp)
                # temp = (temp/255)
                temp = temp.reshape(1,28,28)
                temp = torch.from_numpy(temp)
                total_loss = total_loss + self.mse_loss(x,temp)
                output_image = x.to(dtype=torch.float32)
                temp = temp.to(dtype=torch.float32)
                # print(ssim(output_image.permute(1,2,0).detach().cpu().numpy(),temp.permute(1,2,0).detach().cpu().numpy(), data_range=temp.numpy().max()-temp.numpy().min(),win_size=11,channel_axis=2))
                total_ssim = total_ssim + ssim(temp.detach().cpu().numpy(),output_image.detach().cpu().numpy(), data_range=output_image.detach().cpu().numpy().max()-output_image.detach().cpu().numpy().min(),win_size=11,channel_axis=0)
            # avg_loss_image = avg_loss_image + (total_loss/len(label_image_paths))
            # avg_ssim_image = avg_ssim_image + (total_ssim/len(label_image_paths))
            avg_loss_image = avg_loss_image + (total_loss/50)
            avg_ssim_image = avg_ssim_image + (total_ssim/50)

            # print(avg_ssim_image,'  ',type(avg_ssim_image))
        avg_loss_image = avg_loss_image/BATCH_SIZE
        avg_ssim_image = avg_ssim_image/BATCH_SIZE

        return avg_loss_image-avg_ssim_image,avg_ssim_image

class VAELossFn(nn.Module):
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    def __init__(self) -> None:
        super(VAELossFn,self).__init__()
    
    def forward(self,x_, x_hat_, mean_, log_var_):
        total_loss = 0.0
        total_ssim = 0.0
        avg_loss_image = 0.0
        avg_ssim_image = 0.0
        for x,x_hat,mean,log_var in zip(x_,x_hat_,mean_,log_var_):
            reproduction_loss = nn.functional.binary_cross_entropy((x_hat/255), (x/255), reduction='sum')
            KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
            total_loss = total_loss + reproduction_loss + KLD
            total_ssim = total_ssim + ssim(x.detach().cpu().numpy(),x_hat.detach().cpu().numpy(), data_range=x_hat.detach().numpy().max()-x_hat.detach().numpy().min(),win_size=11,channel_axis=0)
        avg_loss_image = total_loss/BATCH_SIZE
        avg_ssim_image = total_ssim/BATCH_SIZE

        return avg_loss_image-avg_ssim_image,avg_ssim_image

def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    return list(E.parameters()) + list(D.parameters())

class TSNE_plot():
  def plot_fxn(self, embedding_list,n_comp,epoch_number,name):
    # super(TSNE_plot,self)._init_()
    embedding_list = embedding_list.detach().numpy()
    p=30
    # Reshape data to 2D for t-SNE
    data_reshaped = embedding_list.reshape(embedding_list.shape[0], -1)
    if(embedding_list.shape[0]<30):
      p=2
    # Apply t-SNE to reduce dimensionality to 3
    tsne = TSNE(n_components=n_comp, random_state=0,perplexity=p)
    data_embedded = tsne.fit_transform(data_reshaped)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    ax.scatter(data_embedded[:, 0], data_embedded[:, 1], data_embedded[:, 2])

    # Set labels and title
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.title("3D t-SNE Plot")
    plt.savefig('{}_epoch_{}.png'.format(name,epoch_number))
    # plt.show()

class AETrainer:
    """
    Write code for training AutoEncoder here.
    AETrainer(Data,
              E,
              D,
              L[0],
              O,
              A.gpu)
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self,
                 Dataloader_object,
                 Encoder_object,
                 Decoder_object,
                 loss_function,
                 optimizer_object,
                 gpu):
        super(AETrainer,self).__init__()
        self.Dataloader_object = Dataloader_object
        self.Encoder_object = Encoder_object
        self.Decoder_object = Decoder_object
        self.loss_function = loss_function
        self.optimizer_object = optimizer_object
        self.gpu = gpu
        self.tsne_plot = TSNE_plot()

        for epoch in range(0,EPOCH):
            running_loss = 0.0
            running_ssim = 0.0
            epoch_loss = 0.0
            epoch_similarity = 0.0
            embeddings = []
            embeddings = torch.Tensor(embeddings)
            for minibatch,data in tqdm(enumerate(self.Dataloader_object)):
                input_image, label = data['item'], data['label']

                self.optimizer_object.zero_grad()

                output_embedding,_,_,_ = self.Encoder_object(input_image)
                output = self.Decoder_object(output_embedding)

                if(epoch%10 ==0):
                    if(len(embeddings.shape)==1):
                        embeddings = output_embedding
                    embeddings = torch.vstack((embeddings,output_embedding))

                loss,ssim_score = self.loss_function(output,label)
                loss.backward()

                self.optimizer_object.step()

                running_loss +=loss.item()
                epoch_loss +=loss.item()
                running_ssim +=ssim_score
                epoch_similarity +=ssim_score
                if minibatch%10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,running_loss / 10,running_ssim / 10))
                    running_loss = 0.0
                    running_ssim = 0.0
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,epoch_loss/minibatch,epoch_similarity/minibatch))
            if(epoch%10==0):
                self.tsne_plot.plot_fxn(embeddings,3,epoch,'AE')
        torch.save(self.Encoder_object.state_dict(),"AE_Encoder_model.pth")
        torch.save(self.Decoder_object.state_dict(),"AE_Decoder_model.pth")

class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    def __init__(self,
                 Dataloader_object,
                 Encoder_object,
                 Decoder_object,
                 loss_function,
                 optimizer_object,
                 gpu):
        super(VAETrainer,self).__init__()
        self.Dataloader_object = Dataloader_object
        self.Encoder_object = Encoder_object
        self.Decoder_object = Decoder_object
        self.loss_function = loss_function
        self.optimizer_object = optimizer_object
        self.gpu = gpu
        self.tsne_plot = TSNE_plot()

        for epoch in range(0,EPOCH):
            running_loss = 0.0
            running_ssim = 0.0
            epoch_loss = 0.0
            epoch_similarity = 0.0
            embeddings = []
            embeddings = torch.Tensor(embeddings)
            for minibatch,data in tqdm(enumerate(self.Dataloader_object)):
                input_image, label = data['item'], data['label']

                self.optimizer_object.zero_grad()

                output,mean,log_var,output_hat = self.Encoder_object(input_image)
                output = self.Decoder_object(output_hat)

                if(epoch%10 ==0):
                    if(len(embeddings.shape)==1):
                        embeddings = output
                    embeddings = torch.vstack((embeddings,output))

                loss,ssim_score = self.loss_function(input_image,output,mean,log_var)
                loss.backward()

                self.optimizer_object.step()

                running_loss +=loss.item()
                epoch_loss +=loss.item()
                running_ssim +=ssim_score
                epoch_similarity +=ssim_score
                if minibatch%10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,running_loss / 10,running_ssim / 10))
                    running_loss = 0.0
                    running_ssim = 0.0
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,epoch_loss/minibatch,epoch_similarity/minibatch))
            if(epoch%10==0):
                self.tsne_plot.plot_fxn(embeddings,3,epoch,'VAE')
        torch.save(self.Encoder_object.state_dict(),"VAE_Encoder_model.pth")
        torch.save(self.Decoder_object.state_dict(),"VAE_Decoder_model.pth")

class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self,gpu=False):
        self.loaded_encoder_model = Encoder()
        self.loaded_decoder_model = Decoder()
    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

    def from_path(self,sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        self.loaded_encoder_model.load_state_dict(torch.load("AE_Encoder_model.pth"))
        self.loaded_decoder_model.load_state_dict(torch.load("AE_Decoder_model.pth"))

        # sample_image = mpimg.imread(sample)
        # original_image = mpimg.imread(original)
        if len(sample.shape)==3:
            sample = self.rgb2gray(sample)
        if len(original.shape)==3:
            original = self.rgb2gray(original)
        
        sample_image = torch.Tensor(sample)
        sample_image = sample_image.reshape(1,1,28,28)
        original_image = torch.Tensor(original)
        original_for_ssim = original_image.reshape(1,28,28)
        original_image = original_image.reshape(1,1,28,28)

        with torch.no_grad():
            output = self.loaded_encoder_model(sample_image)
            # print(output.dtype)
            output = self.loaded_decoder_model(output[0])
        ssim_score = ssim(original_for_ssim.cpu().numpy(),output[0].detach().cpu().numpy(), data_range=output[0].detach().cpu().numpy().max()-output[0].detach().cpu().numpy().min(),win_size=11,channel_axis=0)
        return ssim_score
    
class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self,gpu=False):
        self.loaded_encoder_model = Encoder()
        self.loaded_decoder_model = Decoder()

    
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    
    def from_path(self,sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        self.loaded_encoder_model.load_state_dict(torch.load("VAE_Encoder_model.pth"))
        self.loaded_decoder_model.load_state_dict(torch.load("VAE_Decoder_model.pth"))

        # sample_image = mpimg.imread(sample)
        # original_image = mpimg.imread(original)
        if len(sample.shape)==3:
            sample = self.rgb2gray(sample)
        if len(original.shape)==3:
            original = self.rgb2gray(original)
        
        sample_image = torch.Tensor(sample)
        sample_image = sample_image.reshape(1,1,28,28)
        original_image = torch.Tensor(original)
        original_for_ssim = original_image.reshape(1,28,28)
        original_image = original_image.reshape(1,1,28,28)

        with torch.no_grad():
            output = self.loaded_encoder_model(sample_image)
            # print(output.dtype)
            output = self.loaded_decoder_model(output[0])        
        ssim_score = ssim(original_for_ssim.cpu().numpy(),output[0].detach().cpu().numpy(), data_range=output[0].detach().cpu().numpy().max()-output[0].detach().cpu().numpy().min(),win_size=11,channel_axis=0)
        return ssim_score

class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()
