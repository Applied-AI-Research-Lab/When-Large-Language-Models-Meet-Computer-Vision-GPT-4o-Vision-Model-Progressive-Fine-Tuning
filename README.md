# When Multimodal Large Language Models Meet Computer Vision: Progressive GPT Fine-Tuning and Stress Testing

## Article
* **Journal**: [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639)
* **Title**: [When Multimodal Large Language Models Meet Computer Vision: Progressive GPT Fine-Tuning and Stress Testing](https://ieeexplore.ieee.org/document/11036181)
* **DOI**: [https://doi.org/10.1109/ACCESS.2025.3579420](https://doi.org/10.1109/ACCESS.2025.3579420)

## Authors
* **Dr. Konstantinos I. Roumeliotis**
* **Prof. Nikolaos D. Tselikas**
* **Prof. Dimitrios K. Nasiopoulos**

## Abstract
The rapid evolution of Multimodal Large Language Models (LLMs) has redefined the landscape of artificial intelligence, with OpenAI’s GPT-4o representing a transformative leap in multimodal learning and processing. This study pioneers the fine-tuning of GPT-4o on image data, addressing a critical gap in evaluating its performance beyond zero-shot capabilities. It investigates the potential of GPT-4o multimodal LLM for computer vision classification tasks, focusing on e-commerce product categorization through a novel, progressive, few-shot learning fine-tuning framework. Leveraging GPT-4o’s newly available multimodal capabilities, we implemented a four-phase progressive training procedure across varying image resolutions (100, 200, and 400 pixels). Each phase was rigorously evaluated for training duration, computational cost, loss metrics, and validation performance. Predictions were assessed using accuracy, precision, recall, and F1-score metrics while closely monitoring prediction costs and processing time. To establish a benchmark, we replicated the process with ResNet-50 and ConvNeXt-tiny models, applying Bayesian optimization for hyperparameter tuning. We explored a range of batch sizes and learning rates using the Optuna library, conducting 100 trials over a maximum of 50 epochs with early stopping and the successive halving pruner to identify optimal configurations for each model. Although ResNet-50 and ConvNeXt-tiny exhibited lower fine-tuning costs, achieving comparable performance required larger batch sizes and lower learning rates. GPT-4o’s multimodal learning approach achieved 87% accuracy with minimal training—just three epochs and a batch size of one—highlighting its efficiency in few-shot learning with limited labeled data. Our findings highlight key trade-offs between cost-efficiency and performance, offering actionable insights for deploying multimodal Large Language Models in resource-constrained scenarios. GPT-4o’s progressive fine-tuning approach showed promising results in computer vision classification tasks, suggesting its potential as a scalable, multimodal AI tool for real-world multimodal learning and vision-based applications.

## Keywords
Multimodal Large Language Models; Multimodal Classification; Computer Vision Classification; Computer vision; Visualization; Accuracy; Computational modeling; Artificial intelligence; Image classification; Electronic commerce; Large language models; Training; Few shot learning

