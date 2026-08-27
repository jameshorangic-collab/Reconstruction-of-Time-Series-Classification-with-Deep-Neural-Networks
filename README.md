# Reconstruction of Time Series Classification with Deep Neural Networks

This project is a reconstruction of the Multi-Layer Perceptron (MLP) architecture used in the 2016 paper *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline*. The paper compares the performance of this MLP with several other time-series classification methods using the UCR 2015 archive as its primary benchmark. The model itself is a fully connected neural network with three hidden layers, ReLU activations, dropout regularization, a softmax classifier, and the Adadelta optimizer.

![MLP architecture used in the reconstruction](./Images/mlp_architecture.png)
*MLP architecture recreated based on Wang et al. (2016).*

The purpose of this project was to develop a deeper understanding of how neural networks function, from the forward pass through backpropagation. I chose to reconstruct this particular paper because it implements a relatively simple neural network framework built from components that are fundamental across modern machine learning. By implementing these operations from scratch, I hoped to better understand the mechanisms that underlie systems now used across many different fields. Although the model itself is simple compared with many of today's architectures, reconstructing it represents an important step toward developing a stronger foundation for more advanced work in machine learning.

The paper itself argues for the use of deep neural networks for time-series classification, with particular emphasis on convolutional neural networks (CNNs), such as the fully convolutional network (FCN). Its importance comes from showcasing that simple end-to-end neural networks could achieve strong performance on the time-series benchmarks of the time without requiring extensive engineering or specialized model design. In particular, the FCN and related architectures demonstrated that convolutional networks could learn useful representations directly from raw time-series data. The work helped establish deep learning as a serious approach to time-series classification and influenced much of the research that followed.

The choice to reconstruct the MLP rather than the FCN was motivated by its foundational role in neural networks and its direct connection to many concepts still used throughout modern machine learning. My reconstruction used the NumPy library to implement the operations, optimizer, and data preprocessing prescribed by the original implementation. For further details on the reconstruction methodology, see [`Docs/Methodology.md`](./Docs/Methodology.md).

## How to Run the Reconstruction

To run the program, first install the required dependencies:

```bash
pip3 install numpy
```

Next, access the UCR time-series dataset archive at https://www.cs.ucr.edu/~eamonn/time_series_data/ and follow the instructions provided on the site to download the dataset.

The scripts read the files specified by `trainFile` and `testFile`. The included Adiac example uses `Adiac_TRAIN.txt` and `Adiac_TEST.txt`. Place the selected dataset files in the project directory or update these variables to point to their actual locations. You must also update the number of classes and input length to match the selected dataset. The current label encoding assumes that class labels are contiguous, one-indexed integers from `1` through `numberOfClasses`; datasets that use a different labeling scheme should have their labels remapped before training or testing.

To run the faithful reconstruction with dropout:

```bash
python3 Train_With_Dropout.py
```

To run the controlled no-dropout ablation instead:

```bash
python3 Train_Without_Dropout.py
```

After training, make sure `weightsFile` in `Test.py` matches the weights file produced by the training script you used, then evaluate the model with:

```bash
python3 Test.py
```

This project took an unexpected turn when I began testing the first dataset, Adiac, and found that my result was substantially lower than the accuracy reported in the original paper (75.2%). After multiple iterations and inspections of my implementation, I was unable to identify a coding error that explained the discrepancy. This led me to investigate whether the difference could instead be caused by another reproducibility issue, such as historical framework differences or a misinterpretation of an implementation detail.

During this investigation, I found a later independent implementation by Fawaz et al., which evaluated several deep-learning architectures across the UCR/UEA time-series archive. Their MLP achieved an average accuracy of 39.7% on Adiac, which was much closer to my faithful reconstruction than to the 75.2% reported by Wang et al.

As a controlled ablation, I also trained the Adiac model without dropout. This increased the mean test accuracy from 36.343% in the faithful reconstruction to 68.363%, substantially closer to the original reported result. This suggests that dropout or related training-configuration differences could have contributed to the discrepancy, although the no-dropout model is not the faithful architecture and does not establish the cause by itself.

To determine whether my implementation was generally flawed or whether Adiac was an unusual case, I also tested additional datasets. On CricketY, my reconstruction achieved 59.231%, compared with 60.0% in the Fawaz implementation and 59.5% in Wang et al., providing a strong reproduction of the expected result. FacesUCR was also reasonably close, although not identical. Together, these results suggest that the large discrepancy is concentrated around Adiac rather than reflecting a general failure of the reconstruction.

For this reason, I interpret Adiac as a reproducibility gap rather than evidence of a specific mistake in either implementation. Possible explanations include differences in historical Keras/TensorFlow behavior, random-number generation, numerical precision, optimizer behavior, or other undocumented implementation details. Full multi-seed results are available in [`Docs/Results.md`](./Docs/Results.md).

Overall, this project provided me with a deeper understanding of how a neural network operates from both a mathematical and programmatic perspective.

## References

- Wang et al., *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline*: https://arxiv.org/abs/1611.06455
- Original Wang et al. implementation: https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline
- Fawaz et al. time-series classification implementation: https://github.com/hfawaz/dl-4-tsc
- UCR Time Series Classification Archive: https://www.cs.ucr.edu/~eamonn/time_series_data/
