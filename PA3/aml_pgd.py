from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
alpha = 5e-4
num_iter = 1500
num_test = 100

pretrained_model = "lenet_mnist_model.pth"
use_cuda=True

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    counter = 0
    # Loop over all examples in test set
    for image, target in test_loader:

        counter += 1
        # print(epsilon, counter) # for debugging purpose
        if counter > num_test:
            break

        # Send the image and label to the device
        image, target = image.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        image.requires_grad = True

        # Forward pass the image through the model
        output = model(image)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Initialize perturbed image
        delta = torch.zeros_like(image)
        # Initialize perturbed image
        # In iteration 0, the perturbed image is the same as the original image
        # But we need create a new tensor in pytorch and only copy the data
        perturbed_image = torch.zeros_like(image, requires_grad=True)
        perturbed_image.data = image.detach() + delta.detach()

        for i in range(num_iter):

            # one iteration of projected gradient descent

            # !! Put your code below

            # Send the perturbed image in the last iteration to model to get output
            temp_output = model(perturbed_image)
            # Calculate the loss given the new output and the target
            loss = F.nll_loss(temp_output, target)
            # Zero all existing gradients
            model.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect gradient w.r.t. the input (perturbed_image)
            gradx = perturbed_image.grad.data
            # Update delta based on the gradient
            delta = perturbed_image + alpha * gradx.sign() - image
            # Adding clipping to maintain [-epsilon,epsilon] range for delta, you can use function torch.clamp
            delta = torch.clamp(delta ,-epsilon, epsilon)
            # Adjust delta to make sure the perturbed image is in the range [0,1] You can apply torch clamp to
            # delta+image, and then update delta as the difference between the clamped image and the original image
            delta = torch.clamp(delta + image, 0, 1)
            # update perturbed_image
            perturbed_image.data = delta
            # Reset gradient of perturbed_image to zero, you can use x.grad.zero_()
            perturbed_image.grad.zero_()

            # !! Put your code above

        # Re-classify the perturbed image
        perturbed_output = model(perturbed_image)

        # Check for success
        final_pred = perturbed_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # print('success!') # for debugging purpose
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(num_test)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, num_test, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
print("Start")
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

fig1 = plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.array(epsilons))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

fig1.savefig('aml_pgd_acc.png')

# Plot several examples of adversarial samples at each epsilon
cnt = 0
fig2 = plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),5,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

fig2.savefig('aml_pgd_ex.png')