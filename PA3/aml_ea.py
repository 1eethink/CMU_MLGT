from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [.1, .2, .3]
num_iter = 500
num_test = 50

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

# EA attack code
def ea_attack(N, image, target_class, epsilon, rho_min, beta_min, num_iter, model, device):
    # perturbed_image = image.detach().clone() # for debugging purpose

    # get image size
    dims = list(image.size())

    # !! Put your code below

    # initialize the population with random images in the feasible region
    # if you are familiar with pytorch and tensor operation, you can create a tensor for the population like the following
    population = torch.empty([N] + dims[1:], device=device).uniform_(-epsilon, epsilon)
    population = torch.clamp(population - image, 0, 1) + image
    # if you prefer to use a list, you may consider the following
    # population = []
    # for n in range(N):
    #     rand_image = torch.empty(dims, device=device).uniform_(-epsilon, epsilon)
    #     rand_image = torch.clamp(rand_image + image, 0, 1) - image
    #     population.append(rand_image)

    # initialize two parameters rho=0.5 and beta=0.4
    rho = 0.5
    beta = 0.4
    # initialize num_plateaus to be 0
    num_plateaus = 0
    
    
    
    for i in range(num_iter):
        new_population = []
        # For each member in the current population, compute the fitness score. Note that you will need to clamp the
        # value to a large range, e.g., [-1000,1000] to avoid getting "inf"
        with torch.no_grad():
            temp_output = model(population)
            fitness = temp_output[:, target_class] - torch.log(-torch.expm1(temp_output[:, target_class]))
            fitness = torch.clamp(fitness, -1000, 1000)
        # Find the elite member, which is the one with the highest fitness score
        elite = population[fitness.argmax()]
        best_fitness = torch.max(fitness)
        
        if i == 0:
            last_best = best_fitness
        if last_best < best_fitness:
            last_best = best_fitness
            
        # Add the elite member to the new population
        new_population.append(elite)
        # If the elite member can succeed in attack, terminate and return the elite member
        if temp_output[fitness.argmax()].argmax().item() == target_class:
            return elite
        # If the elite member’s fitness score is no better than the last population’s elite member’s fitness score,
        # increment num_plateaus. It is recommended to use a threshold of 1e-5 to avoid numerical instability
        if abs(best_fitness - last_best):
            num_plateaus += 1
        else:
            num_plateaus = 0

        for j in range(N-1):
            # Compute the probability each member in the population should be chosen by applying softmax to the fitness
            # scores
            fit_softmax = F.softmax(fitness, dim=0)
            # Choose a member in the current population according to the probability, name it parent_1
            parent_1 = torch.distributions.categorical.Categorical(probs=fit_softmax).sample()
            # Choose a member in the current population according to the probability, name it parent_2
            parent_2 = torch.distributions.categorical.Categorical(probs=fit_softmax).sample()
            # Implemented for the bonus task
            # while parent_1 == parent_2:
            #   parent_2 = torch.distributions.categorical.Categorical(probs=fit_softmax).sample()
            # Generate a “child” image from parent1 and parent2: For each pixel, take parent1’s corresponding pixel
            # value with probability p=fitness(parent1)/(fitness(parent1)+fitness(parent2))
            # and take parent2’s corresponding pixel value with probability 1-p
            p = fit_softmax[parent_1] / (fit_softmax[parent_1] + fit_softmax[parent_2])
            mask = torch.empty_like(population[parent_1]).bernoulli_(p)
            child = population[parent_1] * mask + (1 - mask) * population[parent_2]
            # With probability rho, add a random noise to the children image with pixel-wise value uniformly sampled from
            # [-beta*epsilon,beta*epsilon]
            random_noise = torch.empty_like(child).uniform_(-beta * epsilon, beta * epsilon)
            mask = torch.empty_like(child).bernoulli_(rho)
            child = child + mask * random_noise
            # Apply clipping on the child image to make sure it is in the feasible region F
            child = torch.clamp(child - image.squeeze(0), -epsilon, epsilon) + image.squeeze(0)
            # Add this child to the population, repeat generating children in this way until the population has N members
            new_population.append(child)

        
        population = torch.stack(new_population)
        perturbed_image = elite
         
        # Update the value of rho as max(rho_min,0.5*0.9^num_plateaus)
        rho = max(rho_min, 0.5*(0.9**num_plateaus))
        # Update the value of beta as max(beta_min,0.4*0.9^num_plateaus)
        beta = max(beta_min, 0.4*(0.9**num_plateaus))
        # !! Put your code above
        
    # Return the perturbed image
    return perturbed_image

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

        target_class = (target.item() + 1) % 10

        perturbed_image = ea_attack(10, image, target_class, epsilon, 0.1, 0.15, num_iter, model, device)

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
fig1.savefig('aml_ea_acc.png')

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

fig2.savefig('aml_ea_ex.png')