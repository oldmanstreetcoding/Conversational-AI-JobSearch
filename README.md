
# Conversational AI for Job Search - Fine-Tuning Phi-3.5 Model

## Overview
This project aims to develop and fine-tune a conversational AI model capable of responding to job-related queries, such as skills, location, salary, and experience. The project uses an open-source language model (Phi-3.5-mini-instruct) and fine-tunes it using Low-Rank Adaptation (LoRA) to optimize its ability to handle complex job search queries efficiently. The system is evaluated on various industry-standard metrics, including BLEU, RoGUE, and METEOR, to ensure relevance and fluency.

## Objective
The main objective of the project is to enhance the base Phi-3.5-mini-instruct model to generate specific, accurate, and relevant responses to job-related queries while minimizing computational overhead using LoRA. This project also explores an alternative solution to job search queries using structured data and SQL query generation.

---

## Dataset Information
The dataset consists of job descriptions, resumes, and related information. The key columns include:

- Resume: Candidate's qualifications, skills, and experience
- Job Title: The role's title (e.g., "Senior Frontend Developer")
- Location: Job location
- Salary: Compensation range for the role
- Description: Detailed job responsibilities and requirements
- Handling Missing Data: Missing salary data was handled via imputation based on job title and location, and any remaining missing values were marked as "Unknown."

---

## Project Structure

```
Conversational-AI-JobSearch/
├── code/
│   ├── main_job_search.py               # Main script to run the project
│   ├── preprocessing.py                 # Data loading and preprocessing
│   ├── evaluation.py                    # Model evaluation and comparison
│   ├── job_search_notebook.rar          # Jupyter Notebook for project
│   ├── model_selection.py               # Model base setup
│   ├── finetuning.py                    # Finetuning process
│   ├── requirements.txt                 # Python dependencies
├── visualizations/                      # Model performance visualizations
│   ├── metric_comparison.png            # Visual comparison across metrics
|   |........................
├── README.md                            # Project overview and instructions
├── report_job_search.pdf                # Comprehensive project report
├── updated_slide_deck.pdf               # Project presentation slide

```
---

## How to Run

**Install Dependencies**
Ensure that all required Python libraries are installed. Run the following command:

```
pip install -r code/requirements.txt
```

**Run the Main Script**
You can run the project through main_job_search.py, which provides an interactive menu for different tasks. To run, execute the following command:

```
python code/main_job_search.py
```

**Command-Line Interface**
Upon running the script, you will see the following menu:

```
== Welcome To Conversational AI for Job Search ==
Select Which Step You Want to Proceed:
1. Data Preprocessing
2. Data Preprocessing with Summarization Process for Resume & Description Columns
3. Fine-tune the Phi-3 model on <Job Query, Job Description> pairs
4. Handle user queries for job searches based on different criteria (skills, location, salary)
5. Evaluate the model using ARES, RoGUE, BLEU, and METEOR metrics
6. Exit
== Welcome To Conversational AI for Job Search ==

Enter your choice (1-6):
```

Notes:
- Option 1: Data cleaning and preprocessing without summarization
- Option 2: Data cleaning and preprocessing with summarization
- Option 3: Fine-Tune Model (about 1 hour)
- Option 4: User interaction
- Option 5: Evaluate model and produce visualization
- Option 6: Exit the program

---

## Model Development
Two key steps in model development:

- Preprocessing: Data is cleaned, missing values are handled, and summaries are generated for long resume and description columns.
- Fine-Tuning: LoRA is applied to fine-tune the model for job-related queries, ensuring minimal computational overhead while achieving high accuracy and relevance.

**Fine-Tuning and Evaluation**
You can select Option 3 for fine-tuning the model using LoRA and Option 5 for evaluating the model performance based on job search queries. The results are saved as CSV files and visualizations are stored in the visualizations/ folder.

## Fine-Tuning Setup
Model: Phi-3.5-mini-instruct
Fine-Tuning Technique: LoRA (Low-Rank Adaptation)
Hyperparameters:
Adaptation Dimension: r=16
Batch Size: 4
Gradient Accumulation Steps: 8
Max Sequence Length: 2048
Learning Rate: 5e-5

---

## Model Evaluation
The model is evaluated across skills, location, salary, and experience-based queries using ARES, BLEU, RoGUE, and METEOR metrics.

### Latency and Response Time:

Fine-Tuned Model: Average response time is 2.7 seconds
Base Model: Average response time is 2.9 seconds

### Evaluation Metrics:
BERTScore: For semantic similarity
RoGUE: For content overlap
BLEU: For fluency of generated responses
METEOR: For capturing semantic richness and flexibility in word choice
Example Evaluation Output:

```
>> Accuracy: 0.81
>> ROUGE-L F1: 0.73
>> BLEU Score: 0.60
>> METEOR Score: 0.71
```

---

## Citation
Starter code and guidance were provided during the class lectures, helping streamline the development process.

---

## License
This project is licensed under the MIT License.

---

## Requirements (requirements.txt)

```
time
bert_score
rouge_score
evaluate
torch
unsloth
pandas
numpy
datasets
transformers
matplotlib
os
re
math
sklearn
peft
trl
```
