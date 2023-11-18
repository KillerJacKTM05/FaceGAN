# New Version 
I have removed celeba dataset and implemented RAFDB instead. RAFDB is both easy to use and also more close to my requirements from this GAN project.
I have implemented the first version of training phase based on StarGANv2's generator and discriminator shape. However, i didn't applied AdaIN and Weserstein features yet due to the complexity of actual StarGANv2 project.
I have concluded an initial test training with 20 epoch, batch size of 16. According to that, it seem it's is learning and i have added some results to "GeneratorGanResults" folder.
New features, more stable training and more flexible save settings will be added.

# 25.10 progression
More balanced approach and more insightful image creating loop added. Another 20 epoch training with the batch size of 16 has been completed.
In models, there is not much change instead of some hyperparameter tuning and a usage tring of L2 regularization method.
In epoch 20, we can clearly see gestures in each type of expression, however images are still way more far from being a real person.
Image postprocess and more extensive regularization technics (L1-L2 together, otehr normalization methods etc.) planned to use later.

# 18.11 progression
I couldn't made the standard cGan better than previous iteration. It actually learns the some kind face attributes after 15-ish epochs but cannot have power to generate realistic images.
Instead of continuing, i simply started to a new project which uses domain transfering method. It uses celeb-a and rafdb datasets and tries to merge in together to a single combined model.
Base codes provided. Not yet tested.