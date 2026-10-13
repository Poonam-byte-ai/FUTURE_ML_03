
# Resume and Candidate Screening System

## Project Overview
This project uses Natural Language Processing (NLP) and TF-IDF to compare candidate resumes with a job description and rank candidates by text similarity.

## Objectives
- Process and clean resume text.
- Convert text into numerical features using TF-IDF.
- Compare resumes against a job description using cosine similarity.
- Rank candidates by relevance score.
- Export screening results and a comparison chart.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Matplotlib

## How It Works
1. Load a job description.
2. Read candidate resumes in TXT format.
3. Clean and preprocess the text.
4. Convert the text into TF-IDF vectors.
5. Calculate cosine similarity between each resume and the job description.
6. Rank candidates and save the results.

## Test Results
The project was tested with five sample candidate profiles and one resume supplied for testing.

The resume used for testing received a text similarity score of approximately 16.64% against the sample Junior Machine Learning Engineer job description.

Similarity scores are text-matching measures, not probabilities of hiring success or assessments of a candidate's overall ability.

## Output Files
The outputs folder contains:
- candidate_screening_results.csv
- candidate_relevance_chart.png
- screening_report.txt

## How to Run

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib
```

Run the application:

```bash
python resume_screening.py
```

Place candidate resume text files inside the `resumes` folder. Keep `job_description.txt` in the project root directory.

## Limitations
- The system compares text rather than understanding the full context of a candidate's experience.
- Results depend on the job description and the words used in each resume.
- The similarity score should not be used as the sole basis for hiring decisions.
- The current demonstration uses sample candidate profiles.
