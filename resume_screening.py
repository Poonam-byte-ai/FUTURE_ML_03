
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create output folder
os.makedirs("outputs", exist_ok=True)


# 1. Clean resume and job description text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. Read job description
if not os.path.exists("job_description.txt"):
    raise FileNotFoundError(
        "Create job_description.txt in the project folder first."
    )

with open("job_description.txt", "r", encoding="utf-8") as file:
    job_description = clean_text(file.read())


# 3. Create sample resumes for testing
# Replace these with real resume text files when available.
sample_resumes = {
    "Candidate_01": """
    Junior Machine Learning Engineer. Python, machine learning,
    pandas, NumPy, scikit-learn, SQL, data preprocessing,
    data analysis, model evaluation and visualization.
    Experience developing and evaluating machine learning models.
    """,

    "Candidate_02": """
    Web Developer. HTML, CSS, JavaScript, React, Node.js,
    Express, website development, responsive design and APIs.
    """,

    "Candidate_03": """
    Data Analyst. Python, SQL, pandas, NumPy, data cleaning,
    data analysis, statistics, data visualization and reporting.
    """,

    "Candidate_04": """
    Software Developer. Java, C++, object-oriented programming,
    data structures, algorithms, debugging and software testing.
    """,

    "Candidate_05": """
    Machine Learning Intern. Python, scikit-learn, pandas,
    NumPy, classification, regression, model evaluation,
    feature engineering and data preprocessing.
    """
}


# 4. Load additional resumes from the resumes folder, if available
resume_folder = "resumes"
os.makedirs(resume_folder, exist_ok=True)

for filename in os.listdir(resume_folder):
    if filename.lower().endswith(".txt"):
        filepath = os.path.join(resume_folder, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            candidate_name = os.path.splitext(filename)[0]
            sample_resumes[candidate_name] = file.read()


# 5. Prepare text for comparison
candidate_names = list(sample_resumes.keys())

resume_texts = [
    clean_text(sample_resumes[name])
    for name in candidate_names
]

all_texts = [job_description] + resume_texts


# 6. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(all_texts)

job_vector = tfidf_matrix[0:1]
resume_vectors = tfidf_matrix[1:]


# 7. Calculate relevance scores
similarities = cosine_similarity(
    resume_vectors,
    job_vector
).flatten()

results = pd.DataFrame({
    "Candidate": candidate_names,
    "Relevance_Score_Percent": similarities * 100
})

results = results.sort_values(
    by="Relevance_Score_Percent",
    ascending=False
).reset_index(drop=True)

results.insert(0, "Rank", range(1, len(results) + 1))


# 8. Display and save results
print("\n===== RESUME SCREENING RESULTS =====")
print(results.to_string(index=False))

results.to_csv(
    "outputs/candidate_screening_results.csv",
    index=False
)


# 9. Plot candidate relevance scores
plt.figure(figsize=(10, 6))

plt.bar(
    results["Candidate"],
    results["Relevance_Score_Percent"]
)

plt.title("Resume Relevance to Job Description")
plt.xlabel("Candidate")
plt.ylabel("Text Similarity Score (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/candidate_relevance_chart.png",
    dpi=150
)

plt.close()


# 10. Save a text report
with open(
    "outputs/screening_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("RESUME / CANDIDATE SCREENING REPORT\n")
    file.write("===================================\n\n")
    file.write("Method: TF-IDF and cosine similarity\n\n")
    file.write(results.to_string(index=False))
    file.write(
        "\n\nNote: Similarity scores measure text similarity only. "
        "They are not probabilities of hiring success and should "
        "not be used as the sole basis for hiring decisions.\n"
    )

print("\nFiles saved in the outputs folder.")
print("Add your own .txt resumes to the resumes folder to test them.")
