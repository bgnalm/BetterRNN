# BetterRNN
This repository is for research around RNNs that use persistent memory modules.
## What are we trying to solve?
Both Transformers and LSTMs have some issues with very long sequences, and we will try to tackle this with "snapshot" memories - 
where we save an input/state, and reuse it for future computations. The main difference between this idea and normal RNNs is that these snapshot memories will truly be saved in memory, and not implemented as a recurent cell that receives its hidden state at every state. 

One important issue that we will need to tackle is how to decide which "snapshots" do we need to save. Since we don't want to use LSTM's mechanism, which is costly and often doesn't work well with far connections between two inputs. 
The way we want to solve that is by adding another output to the network, one that will train another part of the network that decides which snapshot should be kept. 
This part will be trained by actually trying to forget a snapshot and seeing if it improves the network's performance. 

In theory, our learning algorithm looks like this:

Given an encoder/decoder network N = (E,D), a snapshot memory size m:
1. For each input x:
2. Encode x, i.e. E(x)
3. (Maybe?) Embed E(x) into some "snapshot memory sapce", i.e. something that describes what kind of information is kept.
4. Pass the value from step 2/3 into the new part of the network that has m outputs (probably a softmax). This output says into which snapshot memory this value should be saved (if any). This new part will consider the other "memory embeddings" that are already saved.
5. Run each possible combination of saved information (maybe we can optimize this) into the decoder. i.e. D(E(x), E(all the memories that are saved))
6. Train the Encoder/Decoder based to the output from step 5
7. Train the new part of the network based on the performance of each combination of memories. 
 
