# Weight Decomposed Low-Rank Adaptation (DoRA)
---
Low-rank adaptation (**LoRA**) is a machine learning technique that modifies a pretrained model (e.g., an LLM) to better suit a specific, often smaller, dataset by adjusting only a small, low-rank subset of the model's parameters.

This approach is important because it allows for efficient finetuning of large models on task-specific data significantly reducing the computational cost and time required for finetuning.

In this notebook, we are going to talk about [Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353), which is a new alterative to **LoRA**, which may outperform **LoRA** by a large margin. We are going to implement both **LoRA** and **DoRA** in PyTorch from scratch in this [notebook](https://github.com/sulaiman-shamasna/Weight-Decomposed-Low-Rank-Adaptation-DoRA).

<!-- ### Environment Setup -->
### Theory

**DoRA** can be seen as an improvement or extension of **LoRA** that is built on top of it, and we can now easily adapt some of our previous code to implement **DoRA**. It, however, can be described in two steps, where the first step is to decompose a pretrained weight matrix into a magnitude *vector(m)* and a directional *matrix(V)*. The second step is applyting **LoRA** to the directional matrix *V* and training the magnitude vector *m* separately.

The decomposition into magnitude and directional components is inspired by the mathematical principle that any vector can be represented as the product of its magnitude(a scalar value indicating its length) and its direction (a unit vector indicating its orientation in space).

IMAGE HERE

Illustration of the direction and magnitude of a single vector. For example, if we have a 2D vector [1,2], we can decompose it into a magnitude 2.24 and a directional vector [0.447, 0.894]. Then 2.24 * [0.447, 0.894]=[1,2].

In **DoRA**, we apply the decomposition into magnitude and directinal components to a whole pretrained matrix **W** instead of a vector, where each column (vector) of the weight matrix corresponds to the weights connecting all inputs to a particular output neuron. So the result of decomposing  
**W** is a magnitude vector **m** that represents the scale or length of each column vector in the weight matrix, as illustrated in the figure below.

IMAGE HERE

Illustration of the weight matrix decomposition in **DoRA**. Then, **DoRA** takes the directional matrix **V** and applies standard **LoRA**, for instance:

EQUATION 3 HERE
$$
W' = \frac{m(V + ΔV)}{norm} = \frac{m(W + AB)}{norm}
$$

The normalization, which I abbreviated as norm to not further complicate things in this overview, is based on the weight normalization method proposed in Saliman's and Kingma's [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks paper](https://arxiv.org/abs/1602.07868).

The **DoRA** two-step process(decomposing a pretrained weight matrix and applying **LoRA** to the directional matrix) is further illustrated in the figure from the **DoRA** paper below.

DORA IMAGE HERE

The motivation for developing **DoRA** is based on analyzing and comparing the **LoRA** and full finetuning learning patterns. The **DoRA** authors found that **LoRA** either increases or decreases magnitude and direction updates proportionally but seems to lack the capability to make only subtle directional changes as found in full finetuning. Hence, the researchers propose the decoupling of magnitude and directional components.

In other words, their **DoRA** method aims to apply **LoRA** only to the directional component,  
**V**, while also allowing the magnitude component,  
**m**, to be trained separately.

Introducing the magnitude vector **m** adds  
**0.01** more parameters if **DoRA** is compared to **LoRA**. However, across both LLM and vision transformer benchmarks, they found the **DoRA** even outperforms LoRA if the **DoRA** rank is halved, for instance, when **DoRA** only uses half the parameters of regular **LoRA**, as shown in the performance comparison below.

TABLE HERE

So, it seems that **DoRA** is much more robust to changes in rank. The possibility to successfully use **DoRA** with relatively small ranks makes this method even more parameter-efficient than **LoRA**.

### Implementation of LoRA and DoRA Layers
Previously, we said that we can initialize a pretrained weight  
*W0* with magnitude *m* and directional component *V*. For instance, we have the following equation:

$$
W_{0} = m\frac{V}{||V||_{c}} = ||W||_{c}\frac{W}{||W||_{c}}
$$
Where *||V||*, *c* is the vector-wide norm of  *V*. Then we can write **DoRA** including the **LoRA** weight update *BA* as shown below:

EQUATION TWO
$$
W' = m\frac{V + ΔV }{||V + ΔV||_c} = \frac{(W_{0} + BA)}{||W_{0} + BA||_c}
$$

Here, *ΔV* is the update to the directional component, matrix *V*.

<!-- This is a display math expression:
$$
\int_{a}^{b} f(x) \, dx
$$ -->

**XX DEVICE CODE XX**

In comparison to ```LinearWithLoRAMerged```, class in [LoRA From Scratch](ADD MY LINK). Both classes integrate a ```LoRALayer``` to augment the original linear layer's weights, but **DoRA** adds **weight normalization and adjustment**.

And we can see in the code below ```LinearWithLoRAMerged``` introduces an additional step involving dynamic normalization of the augmented weights. After combining the original weights with the **LoRA-adjusted** weights ```self.linear.weight``` + ```self.lora.alpha``` * ```lora.T``` it calculates the norm of these combined weights across columns(column_norm). Then, it normalizes the combined weights by dividing them by their norms 

$$V = \frac{CombinedWeight}{ColumnNorm}$$

This step ensures taht each column of the combined weight matrix has a unit norm, which can help stabilize the learning process by maintaining the scale of weight updates.

**DoRA** also introduces a learnable vector ```self.m```, which represents the magnitude of each column of the normalized weight matrix. This parameter allows the model to dynamically adjust the scale of each weight vector in the combined weight matrix during training. This additional flexibility can help the model better capture the importance of different features.

To sum up, ```LinearWithDoRAMerged``` extends the comcept of ```LinearWithLoRAMerged``` by incorporating dynamic weight normalization and scaling to improve the training performance.

Steps of this [notebook](ADD URL) are as follows:

- *Install requirments*
- *Ensure **CUDA**'s availability*:
```python
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')

print(DEVICE)
```
- *Implementing *LoRA* and *DoRA* layers, see compete code [here](ADD URL)*.

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()

...
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()

...
class LinearWithDoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
```

- *Define Multilayer Perceptron layer*
```python
class MultilayerPerceptron(nn.Module):
    def __init__(self, num_features, num_hidden_1, num_hidden_2, num_classes):
        super().__init__()
...
```
- *Load and prepare the Dataset*
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE=64

train_dataset=datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset=datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader=DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader=DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```
- *Define custom training loop and evaluation metrics*
```python
import time

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples=0,0
    with torch.no_grad():
        for features, targets in data_loader:
            features=features.view(-1, 28*28).to(device)
            targets=targets.to(device)
            logits=model(features)
            _, predicted_labels=torch.max(logits,1)
            num_examples+=targets.size(0)
            correct_pred+=(predicted_labels==targets).sum()
        return correct_pred.float()/num_examples*100

def train(num_epochs, model, optimizer, train_loader, device):
    start_time=time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features=features.view(-1, 28*28).to(device)
            targets=targets.to(device)
            
            logits=model(features)
            loss=F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
```

- *Freeze the original weights*
```python
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad=False
        else:
            # recursively freeze linear layers in children modules
            freeze_linear_layers(child)

freeze_linear_layers(model_dora)
```
- *Run the training loop*
```python
optimizer_dora=torch.optim.Adam(model_dora.parameters(), lr=learning_rate)
train(num_epochs, model_dora, optimizer_dora, train_loader, DEVICE)
```

### References
- [LoRA from Scratch](https://www.kaggle.com/code/aisuko/lora-from-scratch)
- [DoRA from Scratch](https://www.kaggle.com/code/aisuko/dora-from-scratch)
- [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)




![DoRA](https://raw.githubusercontent.com/sulaiman-shamasna/Weight-Decomposed-Low-Rank-Adaptation-DoRA/main/images/DoRA.PNG)

![Llama7B](https://raw.githubusercontent.com/sulaiman-shamasna/Weight-Decomposed-Low-Rank-Adaptation-DoRA/main/images/Llama7B.PNG)

![matrix_dec](https://raw.githubusercontent.com/sulaiman-shamasna/Weight-Decomposed-Low-Rank-Adaptation-DoRA/main/images/matrix_dec.PNG)

![table](https://raw.githubusercontent.com/sulaiman-shamasna/Weight-Decomposed-Low-Rank-Adaptation-DoRA/main/images/table.PNG)

![vector_decomposition](https://raw.githubusercontent.com/sulaiman-shamasna/Weight-Decomposed-Low-Rank-Adaptation-DoRA/main/images/vector_decomposition.PNG)