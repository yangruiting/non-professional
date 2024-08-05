## FAQ——Frequently Asked Questions

# ***If you have new questions, please describe them here:***

```
                 Your questions

```

# Already have answers
***1. How does `torch.nn.CrossEntropyLoss()` calculate the loss?***

Here is the official documentation: [torch.nn.CrossEntropyLoss()](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

We can think of `nn.CrossEntropyLoss()` as `log_softmax()` + `nn.NLLLose()`.

Assume that the output tensor obtained through a network is _x_ (x_new in our code) and the labels are: _y_ (z_new in our code). The calculation principle is as follows:

$p = softmax(\mathit{x})=\frac{exp(i,j)}{\sum_j exp(i,j)}$

$y=one-hot(y)$

$loss = -\frac{1}{N} \sum_k y_k log(p)$

The purpose of the neural network is to make the approximate distribution p close to the true distribution y through training. The closer the predicted distribution is to the true distribution, the smaller the cross entropy loss is, and the farther the predicted distribution is from the true distribution, the greater the cross entropy loss is.

***2. How can I see the shape of each network layer's data?***

In deep learning, the shape transformation of tensors is common and frequent. The logic behind the tensor transformation is the algorithm of the corresponding layer. In general, you can understand the process of tensor shape transformation based on the algorithm. 

I have annotated the transformed shapes at key steps. We use tensors uniformly. You can use the command `tensor.shape` or `tensor.size()` to view the shape of a tensor in detail. The following are commonly used commands for tensor shape transformation. If you are looking at changes in tensor shape, you can pay special attention to them.
```
tensor.view()  # Change the shape.
tensor.reshape() # Similar to the tensor.view().
tensor.squeeze() # Compress all dimensions whose dimension is 1.
tensor.unsqueeze(dimx) # Dimension upscaling at the dimx dimension
torch.cat([tensor a,tensor b],dimx) # Concatenate tensors at dimx dimension.
tensor.repeat() # Duplicate at a dimension.
tensor[:,-1,:] # tensor slicing
torch.view_as_complex() # dim//2
torch.view_as_real() # d*2
```

***3. What are the units of parameter quantities?***

Commonly used units are: 1K=1,000; 1M=1,000,000; 1B/1G=1,000,000,000.

Conversion between parameter quantity (P) and display memory usage (D): 

The default type of Tensor is single-precision floating point number fp32. So, D=4P/1024/1024.

***4. Why not use einops.rearrange?***
Packages _einops_ and _mmengine_ conflict.

