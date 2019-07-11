# CS490_DomonicNeal

In this lesson we learned about types of ANNs and Recurrent Neural Networks. With this information we learned how to analyze and perform 
word embedding on a given dataset.


Part 1: In the code provided there are three mistake which stop the code to get run successfully; find those mistakes and 
explain why they need to be corrected to be able to get the code run.

  1. Removed input_dim as it was needed after the first layer was created and implemented
  2. We had to perfrom flattening after implementing our embedded layer onto our dataset
  3. Change activiation on output layer to softmax
 
Part 2: Add embedding layer to the model, did you experience any improvement?
  Yes there are imporvements because the embedding layer allows us to perform better analysis on the data of words and determine the
  probability and help the machine learn what words are of relation / close to one another.

Part 3: Apply the code on 20_newsgroup data set we worked in the previous classes.
