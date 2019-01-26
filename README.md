# Machine-Learning-Hard-Margin-and-Soft-Margin-SVM-for-Faces-Dataset

<b>Problem Statement :</b>  Implement Hard Margin and Soft Margin SVM for Face Recognition Dataset. 

<b>Language Used :</b> Python

<b>External Libraries Used :</b>  Numpy, cvxopt, sklearn, scipy

<b>Dataset</b> : Face Recognition dataset

## Instructions to Run :

The program does not require you to do anything. Once the files are extracted, you will need to import all the external libraries in the interpreter. 

Once the libraries are successfully imported, you  can directly run the program in order to see the results. 

I have written the same program to handle all the 3 sub questions. There is only 1 minor thing that needs to be changed in order to test the program. 

Currently, the rate is set to 100. But  you can alter the slack variable by changing the value of  ‘c’ variable on line 14 in the program. 

If the value of c is set to 0, the program becomes a hard margin SVM and if it is kept a positive non-zero value, then it is a soft margin SVM. 

## Comments :

We already know that hard margin SVM does not allow any kind of noise so hard margin SVM gives the best accuracy. 

On the other hand, as we keep increasing the slack in our SVM, the model allows more and more errors in the margins and the model start becoming less accurate. 

After running the code multiple times with many different values of the slack variable, I have come to the conclusion that the hard margin SVM has the best accuracy which comes around 98% and for soft margin SVM, the accuracy stays in the range of 95% and 98% where the accuracy goes on decreasing as we keep increasing the value of the slack variable. 

For the purpose of facial recognition, we can conclude that hard margin SVM is the best method but for different applications, different types of SVMs can be taken into considerations for best performance. 
