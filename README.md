# New Version 
I have removed celeba dataset and implemented RAFDB instead. RAFDB is both easy to use and also more close to my requirements from this GAN project.
I have implemented the first version of training phase based on StarGANv2's generator and discriminator shape. However, i didn't applied AdaIN and Weserstein features yet due to the complexity of actual StarGANv2 project.
I have concluded an initial test training with 20 epoch, batch size of 16. According to that, it seem it's is learning and i have added some results to "GeneratorGanResults" folder.
New features, more stable training and more flexible save settings will be added.