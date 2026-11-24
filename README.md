# ML Solutions

These are notebooks from my machine learning practice, mostly olympiad-style (IOAI prep) problems and home tasks, plus some smaller projects. Most of them use PyTorch and Hugging Face and were run on Colab or Kaggle, so the paths point to Google Drive or Kaggle inputs and the data isn't included here.

## Audio

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [AST_Ridge_ClassIncremental_Audio_Classification](audio/AST_Ridge_ClassIncremental_Audio_Classification.ipynb) | Frozen AST embeddings, waveform augmentation, label repair, class-balanced ridge head | Extends a 16-class AST audio classifier to 29 classes with a ridge head on pooled multi-layer embeddings. | 0.926 task score |
| [AST_LoRA_KL_Distillation_Audio_Classification](audio/AST_LoRA_KL_Distillation_Audio_Classification.ipynb) | AST, LoRA (PEFT), KL distillation, weighted sampling, AMP | Fine-tunes AST to 29 classes with LoRA and distills from the frozen original model to limit forgetting. | 0.873 task score (0.878 local val) |

## NLP

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [Activation_Steering_MiniLM_Sentiment_Flip](nlp/Activation_Steering_MiniLM_Sentiment_Flip.ipynb) | Forward hooks, difference-of-means steering vectors, norm capping | Builds sentiment steering vectors from mean layer activations and injects them into MiniLM to flip a frozen classifier. | 0.810 local flip rate (0.245 pre-judge score) |
| [LoRA_GPT2_SmolTalk_Rewrite_Instruction_Tuning](nlp/LoRA_GPT2_SmolTalk_Rewrite_Instruction_Tuning.ipynb) | GPT-2, LoRA (PEFT), chat special tokens, assistant-only loss masking | Instruction-tunes GPT-2 on the SmolTalk rewrite subset and generates rewrites for new prompts. | - |
| [DistilGPT2_Chatbot_FineTune](nlp/DistilGPT2_Chatbot_FineTune.ipynb) | DistilGPT2 fine-tuning, prompt masking, warmup scheduler, top-k/top-p sampling | Fine-tunes DistilGPT2 on question-answer pairs as a simple chatbot, with loss only on the reply. | - |
| [BERT_Recipe_Cuisine_Classification](nlp/BERT_Recipe_Cuisine_Classification.ipynb) | BERT-base fine-tuning, MLP head, stratified split | Predicts a recipe's cuisine from its ingredient list (What's Cooking dataset). | 0.797 val accuracy |
| [Qwen_LLM_InfoGain_Twenty_Questions_Animal_Deduction](nlp/Qwen_LLM_InfoGain_Twenty_Questions_Animal_Deduction.ipynb) | Qwen2.5-3B yes/no logit margins, KV-cache reuse, Bayesian posterior, information gain | Plays twenty questions against an LLM oracle by picking the most informative question over 1472 animals. | 0.770 dev score (99.3% solved, 11.27 queries) |

## Multimodal

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [CLIP_Flickr8k_Zero_Shot_Text_To_Image_Retrieval](multimodal/CLIP_Flickr8k_Zero_Shot_Text_To_Image_Retrieval.ipynb) | CLIP ViT-B/32, embedding extraction, cosine similarity top-k | Embeds Flickr8k images with CLIP and returns the top 5 images for a text query. | - |
| [CLIP_Multi_Line_Description_Image_Retrieval](multimodal/CLIP_Multi_Line_Description_Image_Retrieval.ipynb) | CLIP ViT-B/16, per-line embeddings, line deduplication, top-k mean similarity | Finds the matching image for a long multi-line description by scoring each line with CLIP. | MRR@10 0.515 (train set) |

## Vision

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [ViT_CIFAR10_Two_Stage_FineTune](vision/ViT_CIFAR10_Two_Stage_FineTune.ipynb) | ViT-B/16, head-then-full fine-tuning, separate LRs, cosine warmup | Fine-tunes ViT on a CIFAR-10 subset, first the head only and then the whole network. | 0.975 val accuracy (1 full epoch) |
| [VAE_ResNet18_MVTec_Bottle_Reconstruction](vision/VAE_ResNet18_MVTec_Bottle_Reconstruction.ipynb) | Conv VAE, pretrained ResNet-18 encoder, KL annealing | Trains a VAE on defect-free MVTec bottle images and compares reconstruction error on train and test images. | - |
| [Conv_Denoising_Autoencoder_CIFAR10](vision/Conv_Denoising_Autoencoder_CIFAR10.ipynb) | Conv denoising autoencoder, Gaussian noise, BCE loss | Trains a small autoencoder to reconstruct clean CIFAR-10 images from noisy ones. | - |
| [UNet_Density_Map_Chicken_Counting](vision/UNet_Density_Map_Chicken_Counting.ipynb) | U-Net decoder on a pretrained feature extractor, density map regression, counting loss | Predicts chicken density maps and counts chickens (IOAI 2025 task). | 0.829 (local eval on training data) |
| [FasterRCNN_MobileNetV3_Dog_Detection_Transfer](vision/FasterRCNN_MobileNetV3_Dog_Detection_Transfer.ipynb) | Faster R-CNN MobileNetV3-FPN, frozen backbone, YOLO box conversion, box-aware augmentation | Adapts a pretrained cat detector to detect dogs by fine-tuning the detection heads. | - |
| [CNN_Sign_Language_MNIST](vision/CNN_Sign_Language_MNIST.ipynb) | CNN with BatchNorm and Dropout, Adam, confusion matrix | Trains a CNN on Sign Language MNIST and plots accuracy curves and a confusion matrix. | - |

## Imitation learning

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [FiLM_ResNet_Behavior_Cloning_Grid_Delivery_Robot](rl-imitation/FiLM_ResNet_Behavior_Cloning_Grid_Delivery_Robot.ipynb) | Behavior cloning, FiLM ResNet, D4 symmetry augmentation, multi-step action loss, no-backtrack decoding | Trains a policy from expert demos for a delivery robot on an 8x8 grid. | 0.994 success rate (1600 scenarios) |
| [FiLM_FullyConv_Behavior_Cloning_Variable_Size_Grids](rl-imitation/FiLM_FullyConv_Behavior_Cloning_Variable_Size_Grids.ipynb) | Behavior cloning, fully convolutional FiLM ResNet, readout at agent position, D4 augmentation | Same delivery robot task, but the network works on grids of any size. | 0.934 success rate (mock test) |

## Classical ML

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [NumPy_Linear_Regression_From_Scratch](classical-ml/NumPy_Linear_Regression_From_Scratch.ipynb) | Gradient descent in NumPy, sklearn comparison, polynomial features, RMSE/MAE/MAPE | Implements linear regression from scratch and looks at outliers, metrics and overfitting. | - |
| [Logistic_Regression_Heart_Disease](classical-ml/Logistic_Regression_Heart_Disease.ipynb) | Categorical encoding, polynomial features, StandardScaler, LogisticRegression | Fits a logistic regression on the 2020 CDC heart disease survey data. | 0.92 accuracy (positive-class F1 0.18) |
| [LogReg_vs_MLP_Pima_Diabetes_Missing_Values](classical-ml/LogReg_vs_MLP_Pima_Diabetes_Missing_Values.ipynb) | Missing value injection, imputation, LogisticRegression, PyTorch MLP | Makes a Pima diabetes task with missing values and compares logistic regression with a small MLP. | 0.779 test accuracy (logistic regression) |

## Python basics

| Notebook | Techniques | What it does | Result |
|---|---|---|---|
| [Python_Fundamentals_Exercises](python-basics/Python_Fundamentals_Exercises.ipynb) | Statistics from scratch, file I/O, list comprehensions, binary search | Python homework exercises, with the task statements in Georgian. | - |

Some notebooks still have rough edges (interrupted runs, leftover experiments). I kept them close to how I actually worked on them.
