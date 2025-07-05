# Task 1: Exploratory Data Analysis and Preprocessing

## Overview

CrediTrust Financial receives a high volume of unstructured customer complaints across multiple financial product lines. This task focuses on exploring and preparing the Consumer Financial Protection Bureau (CFPB) complaint dataset for use in a Retrieval-Augmented Generation (RAG) system. The primary objective is to filter, clean, and normalize the complaint narratives to ensure quality inputs for semantic search and large language model (LLM) processing.

---

## Objectives

- Load and inspect the raw complaint dataset.
- Filter complaints to focus on five key product categories:
  - Credit Card
  - Personal Loan
  - Buy Now, Pay Later (BNPL)
  - Savings Account
  - Money Transfers
- Remove entries without complaint narratives.
- Clean and standardize the narrative text.
- Perform exploratory data analysis to understand the distribution, quality, and structure of the data.
- Save the cleaned dataset for use in downstream tasks (chunking, embedding, vector indexing).

---

## Dataset Summary

- Source: Consumer Financial Protection Bureau (CFPB)
- Input File: data/complaints.csv
- Filtered Output: data/filtered_complaints.csv
- Key Column: Consumer complaint narrative

---

## Preprocessing Details

The following steps were implemented to clean and prepare the dataset:

1. Product Filtering:  
   Selected only complaints related to the five target products. Variations and inconsistencies in product labels were normalized using a custom mapping strategy.

2. Narrative Cleaning:  
   - Converted text to lowercase  
   - Removed boilerplate phrases (e.g., “I am writing to file a complaint...”)  
   - Removed special characters and excess whitespace

3. Handling Missing Data:  
   - Dropped records without narrative text  
   - Ensured all remaining narratives are suitable for embedding

---

## Exploratory Analysis

- Total complaints after filtering: 454,472
- Complaints with <10 words: 2,125 (potentially too short for analysis)
- Complaints with >500 words: 30,968 (will require chunking before embedding)

Visualizations included:

- A bar chart showing the distribution of complaints across normalized product categories
- A histogram illustrating the distribution of narrative word counts (focused on 0–500 words)

---

## Output Files

- data/filtered_complaints.csv: Cleaned and filtered complaint dataset
- notebooks/01_eda_preprocessing.ipynb: EDA and preprocessing notebook

---

## Next Steps

Proceed to Task 2: Chunking, Embedding, and Vector Indexing, where the cleaned narratives will be processed into semantically searchable chunks and stored in a vector database (FAISS or ChromaDB).

---